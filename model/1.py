import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import GATConv, global_mean_pool
import matplotlib.pyplot as plt

# Channel names for heatmap axes
CHANNEL_NAMES = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
                 'T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']

###########################################
# 1D-модули на основе EEGNet для временной обработки
###########################################

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size, padding=padding,
            groups=in_channels, bias=bias
        )

    def forward(self, x):
        return self.depthwise(x)


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size, padding=padding,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)


class EEGFeatureExtractor1D(nn.Module):
    """
    Экстрактор временных признаков для каждого электрода (EEGNet‑1D).
    Вход:  [B*n_channels, 1, n_samples]
    Выход: [B*n_channels, feat_dim]
    """
    def __init__(self,
                 temporal_filters=16,
                 output_dim=16,
                 kernel_size=25,
                 pooling_kernel=4,
                 dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(
            1, temporal_filters,
            kernel_size=kernel_size,
            padding=kernel_size//2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(temporal_filters)
        self.elu = nn.ELU()

        self.depthwise = DepthwiseConv1d(
            temporal_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(temporal_filters)

        self.separable = SeparableConv1d(
            temporal_filters, output_dim,
            kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(output_dim)

        self.pool    = nn.AvgPool1d(pooling_kernel)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu(x)

        x = self.separable(x)
        x = self.bn3(x)
        x = self.elu(x)

        x = self.pool(x)
        x = self.dropout(x)
        x = F.adaptive_avg_pool1d(x, 1)
        return x.squeeze(-1)


###########################################
# Статический граф: вычисление adjacency и edge_index
###########################################

def compute_adjacency(n_channels):
    coords = np.random.rand(n_channels, 2)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    adj = 1.0 / (dist + 1e-5)
    np.fill_diagonal(adj, 0)
    return adj / np.max(adj)


def create_edge_index(adj_matrix, threshold=0.3):
    src, dst = np.where((adj_matrix > threshold) & (np.eye(adj_matrix.shape[0]) == 0))
    edge_index = np.vstack((src, dst))
    return torch.tensor(edge_index, dtype=torch.long)


###########################################
# SE‑блок для графовых признаков
###########################################

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        # x: [N, C]
        se = x.mean(dim=0, keepdim=True)       # [1, C]
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se


###########################################
# Оптимизированная модель с визуализацией attention
###########################################

class Optimized_DSTGAT_Attn(nn.Module):
    """
    EEGNet‑GAT с возвратом attention‑весов для визуализации.
    """
    def __init__(self,
                 n_channels,
                 eeg_feat_dim=16,
                 gcn_channels=32,
                 num_classes=2,
                 dropout_rate=0.5,
                 gat_heads=4):
        super().__init__()
        self.n_channels = n_channels
        self.feature_extractor = EEGFeatureExtractor1D(
            temporal_filters=16,
            output_dim=eeg_feat_dim,
            kernel_size=25,
            pooling_kernel=4,
            dropout_rate=dropout_rate
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.gat1 = GATConv(
            eeg_feat_dim, gcn_channels,
            heads=gat_heads, concat=False, dropout=dropout_rate
        )
        self.gat1_bn = nn.BatchNorm1d(gcn_channels)
        self.skip1   = nn.Linear(eeg_feat_dim, gcn_channels)

        self.gat2 = GATConv(
            gcn_channels, gcn_channels,
            heads=gat_heads, concat=False, dropout=dropout_rate
        )
        self.gat2_bn = nn.BatchNorm1d(gcn_channels)
        self.skip2   = nn.Identity()

        self.fc = nn.Linear(gcn_channels, num_classes)

    def forward(self, x, edge_index, return_attention_weights=False):
        B, C, T = x.shape

        # 1) Временной экстрактор
        x = x.view(B*C, 1, T)
        x = self.feature_extractor(x)

        # узлы
        x = x.view(B, C, -1).view(B*C, -1)

        # пакетный граф
        new_ei = []
        for i in range(B):
            new_ei.append(edge_index + i*C)
        new_ei = torch.cat(new_ei, dim=1).to(x.device)
        batch_vec = torch.arange(B, device=x.device).unsqueeze(1).repeat(1, C).view(-1)

        # 2) Первый GAT (возврат alpha)
        if return_attention_weights:
            x1, (ei1, alpha1) = self.gat1(
                x, new_ei, return_attention_weights=True
            )
        else:
            x1 = self.gat1(x, new_ei)
        x1 = self.gat1_bn(x1)
        x1 = F.gelu(x1 + self.skip1(x))
        x1 = self.dropout(x1)

        # 3) Второй GAT
        if return_attention_weights:
            x2, (ei2, alpha2) = self.gat2(
                x1, new_ei, return_attention_weights=True
            )
        else:
            x2 = self.gat2(x1, new_ei)
        x2 = self.gat2_bn(x2)
        x2 = F.gelu(x2 + x1)
        x2 = self.dropout(x2)

        # 4) pooling + логиты
        out = global_mean_pool(x2, batch_vec)
        logits = self.fc(out)

        if return_attention_weights:
            # alpha shape: [E, heads]
            return logits, edge_index, alpha1, alpha2
        return logits

###########################################
# Загрузка данных
###########################################

class EEGDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        for s in os.listdir(root_dir):
            p = os.path.join(root_dir, s)
            if os.path.isdir(p):
                for f in os.listdir(p):
                    if f.endswith('.npz'):
                        self.files.append(os.path.join(p, f))
        self.files.sort()

    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        clip  = torch.tensor(d['clip'], dtype=torch.float32)
        label = torch.tensor(int(d['label']), dtype=torch.long)
        return clip, label

###########################################
# Обучение
###########################################

def train_model(model, train_loader, test_loader, edge_index, device,
                epochs=50, lr=1e-3):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train(); rloss=0; corr=0; tot=0
        for clips, labs in train_loader:
            clips, labs = clips.to(device), labs.to(device)
            opt.zero_grad()
            outs = model(clips, edge_index.to(device))
            loss = criterion(outs, labs)
            loss.backward(); opt.step()
            rloss += loss.item()*labs.size(0)
            preds = outs.argmax(1)
            corr += (preds==labs).sum().item(); tot += labs.size(0)
        train_acc = corr/tot; train_loss = rloss/tot
        model.eval(); rloss=0; corr=0; tot=0
        with torch.no_grad():
            for clips, labs in test_loader:
                clips, labs = clips.to(device), labs.to(device)
                outs = model(clips, edge_index.to(device))
                loss=criterion(outs, labs)
                rloss += loss.item()*labs.size(0)
                preds = outs.argmax(1)
                corr += (preds==labs).sum().item(); tot += labs.size(0)
        test_acc = corr/tot; test_loss = rloss/tot
        sched.step(test_loss)
        print(f"Epoch {ep}/{epochs} | Train {train_loss:.4f},{train_acc:.4f} | Test {test_loss:.4f},{test_acc:.4f}")

###########################################
# Визуализация attention-карт
###########################################

def visualize_attention(model, loader, edge_index, device, n_samples=1):
    model.eval()
    clips, _ = next(iter(loader))
    clips = clips[:n_samples].to(device)
    with torch.no_grad():
        logits, ei, alpha1, _ = model(
            clips, edge_index.to(device), return_attention_weights=True
        )
    # Среднее по головам
    attn = alpha1.mean(dim=1).cpu().numpy()  # [E]
    ei = ei.cpu().numpy()
    C = model.n_channels
    mat = np.zeros((C, C))
    for (s, d), w in zip(ei.T, attn):
        mat[s, d] = w
    # Плот
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(mat, cmap='viridis')
    ax.set_xticks(range(C)); ax.set_yticks(range(C))
    ax.set_xticklabels(CHANNEL_NAMES, rotation=90)
    ax.set_yticklabels(CHANNEL_NAMES)
    fig.colorbar(im, ax=ax)
    ax.set_title('GAT Layer 1 Attention Map')
    plt.tight_layout()
    plt.show()

###########################################
# Основной
###########################################

if __name__ == '__main__':
    data_root = '/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = EEGDataset(data_root)
    n_tot = len(ds)
    n_tr  = int(0.8*n_tot)
    tr, te = random_split(ds, [n_tr, n_tot-n_tr])
    tr_ld = DataLoader(tr, batch_size=16, shuffle=True, drop_last=True)
    te_ld = DataLoader(te, batch_size=16, shuffle=False)

    C = 19
    adj = compute_adjacency(C)
    ei  = create_edge_index(adj, 0.3)

    model = Optimized_DSTGAT_Attn(
        n_channels=C, eeg_feat_dim=16,
        gcn_channels=32, num_classes=2,
        dropout_rate=0.5, gat_heads=4
    ).to(device)

    train_model(model, tr_ld, te_ld, ei, device, epochs=50, lr=1e-3)
    visualize_attention(model, te_ld, ei, device, n_samples=1)
