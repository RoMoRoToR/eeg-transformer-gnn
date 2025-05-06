import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import GATConv, global_mean_pool
import matplotlib.pyplot as plt


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
# Улучшенная модель с динамической топологией
###########################################

class Improved_DSTGAT_Attn_Adaptive(nn.Module):
    """
    Улучшенная DSTGAT‑Attn с сочетанием статического и динамического графа.
    """
    def __init__(self,
                 n_channels,
                 eeg_feat_dim=16,
                 gcn_channels=32,
                 num_classes=2,
                 dropout_rate=0.5,
                 gat_heads=4,
                 dyn_threshold=0.1):
        super().__init__()
        self.n_channels = n_channels

        # Временной экстрактор
        self.feature_extractor = EEGFeatureExtractor1D(
            temporal_filters=16,
            output_dim=eeg_feat_dim,
            kernel_size=25,
            pooling_kernel=4,
            dropout_rate=dropout_rate
        )
        self.dropout = nn.Dropout(dropout_rate)

        # Learnable embeddings для динамической топологии
        self.node_embed = nn.Parameter(torch.randn(n_channels, eeg_feat_dim))

        # GAT‑слои + SE‑блоки + residual
        self.gat1   = GATConv(eeg_feat_dim, gcn_channels,
                              heads=gat_heads, concat=False, dropout=dropout_rate)
        self.bn1    = nn.BatchNorm1d(gcn_channels)
        self.skip1  = nn.Linear(eeg_feat_dim, gcn_channels, bias=False)
        self.se1    = SEBlock(gcn_channels)

        self.gat2   = GATConv(gcn_channels, gcn_channels,
                              heads=gat_heads, concat=False, dropout=dropout_rate)
        self.bn2    = nn.BatchNorm1d(gcn_channels)
        self.skip2  = nn.Identity()
        self.se2    = SEBlock(gcn_channels)

        self.classifier = nn.Linear(gcn_channels, num_classes)
        self.dyn_thresh = dyn_threshold

    def forward(self, x, static_edge_index):
        # x: [B, n_channels, n_samples]
        B, C, T = x.shape

        # 1) Временный экстрактор
        x = x.view(B * C, 1, T)                   # [B*C, 1, T]
        x = self.feature_extractor(x)             # [B*C, feat_dim]

        # 2) Подготовка графа
        # 2a) Статический
        # static_edge_index: [2, E_static] в CPU или нужном device
        # 2b) Динамический (learnable)
        A_dyn = F.relu(self.node_embed @ self.node_embed.T)  # [C, C]
        A_dyn = A_dyn / (A_dyn.sum(dim=1, keepdim=True) + 1e-6)
        src_dyn, dst_dyn = torch.nonzero(A_dyn > self.dyn_thresh,
                                         as_tuple=True)
        dyn_edge_index = torch.stack([src_dyn, dst_dyn], dim=0)

        # Объединяем статические и динамические рёбра
        edge_index_base = torch.cat([static_edge_index, dyn_edge_index], dim=1)

        # 3) Расширение для batch
        # Сдвигаем индексы для каждого примера
        edge_indices = []
        for i in range(B):
            edge_indices.append(edge_index_base + i * C)
        edge_index = torch.cat(edge_indices, dim=1).to(x.device)

        # batch vector: [B*C] -> [0,...,0,1,...,1,...]
        batch_vec = torch.arange(B, device=x.device).unsqueeze(1)\
                        .repeat(1, C).view(-1)

        # 4) Первый GAT‑слой
        x_in = x
        x    = self.gat1(x, edge_index)
        x    = self.bn1(x)
        x    = F.gelu(x + self.skip1(x_in))
        x    = self.se1(x)
        x    = self.dropout(x)

        # 5) Второй GAT‑слой
        x_in2 = x
        x     = self.gat2(x, edge_index)
        x     = self.bn2(x)
        x     = F.gelu(x + x_in2)
        x     = self.se2(x)
        x     = self.dropout(x)

        # 6) Глобальный pooling и классификация
        x = global_mean_pool(x, batch_vec)  # [B, gcn_channels]
        return self.classifier(x)


###########################################
# Класс для загрузки данных EEG
###########################################

class EEGDataset(Dataset):
    def __init__(self, root_dir):
        self.file_paths = []
        for subj in os.listdir(root_dir):
            subj_path = os.path.join(root_dir, subj)
            if os.path.isdir(subj_path):
                for fname in os.listdir(subj_path):
                    if fname.endswith('.npz'):
                        self.file_paths.append(os.path.join(subj_path, fname))
        self.file_paths = sorted(self.file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data  = np.load(self.file_paths[idx])
        clip  = torch.tensor(data['clip'], dtype=torch.float32)
        label = torch.tensor(int(data['label']), dtype=torch.long)
        return clip, label


###########################################
# Обучение и визуализация
###########################################

def train_model(model, train_loader, test_loader, static_edge_index, device,
                num_epochs=50, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train_losses, train_accs = [], []
    test_losses,  test_accs  = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for clips, labels in train_loader:
            clips = clips.to(device)
            labels= labels.to(device)
            optimizer.zero_grad()
            outputs = model(clips, static_edge_index.to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * clips.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for clips, labels in test_loader:
                clips = clips.to(device)
                labels= labels.to(device)
                outputs = model(clips, static_edge_index.to(device))
                loss = criterion(outputs, labels)

                running_loss += loss.item() * clips.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        test_loss = running_loss / total
        test_acc  = correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step(test_loss)
        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train: {train_loss:.4f}, {train_acc:.4f} | "
              f"Test:  {test_loss:.4f}, {test_acc:.4f}")

    return train_losses, train_accs, test_losses, test_accs


def plot_curves(train_losses, train_accs, test_losses, test_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses,  label='Test  Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs,  label='Train Acc')
    plt.plot(epochs, test_accs,   label='Test  Acc')
    plt.xlabel('Epoch'); plt.ylabel('Acc');  plt.legend(); plt.title('Accuracy')
    plt.tight_layout()
    plt.show()


###########################################
# Основной блок
###########################################

if __name__ == "__main__":
    data_root = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EEGDataset(data_root)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test  = n_total - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test])
    train_ld = DataLoader(train_ds, batch_size=16, shuffle=True,  drop_last=True)
    test_ld  = DataLoader(test_ds,  batch_size=16, shuffle=False)

    # статический граф
    n_channels = 19
    adj = compute_adjacency(n_channels)
    static_edge_index = create_edge_index(adj, threshold=0.3)

    # инициализация модели
    model = Improved_DSTGAT_Attn_Adaptive(
        n_channels=n_channels,
        eeg_feat_dim=16,
        gcn_channels=32,
        num_classes=2,
        dropout_rate=0.5,
        gat_heads=4,
        dyn_threshold=0.1
    ).to(device)

    # обучение
    train_losses, train_accs, test_losses, test_accs = train_model(
        model, train_ld, test_ld, static_edge_index, device,
        num_epochs=50, lr=1e-3
    )
    plot_curves(train_losses, train_accs, test_losses, test_accs)

    torch.save(model.state_dict(), "improved_dstgat_attn_adaptive.pth")
    print("Training finished, model saved.")
