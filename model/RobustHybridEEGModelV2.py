import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.nn import GATConv, global_mean_pool
from dropblock import DropBlock2D  # pip install dropblock

###########################################
# Поведенческие аугментации
###########################################
def augment_clip(clip):
    # clip: Tensor [C, T]
    C, T = clip.shape
    # 1) случайный тайм-кроп 90% длины
    crop = int(0.9 * T)
    start = np.random.randint(0, T - crop + 1)
    clip = clip[:, start:start+crop]
    clip = F.interpolate(clip.unsqueeze(0), size=T, mode='linear', align_corners=False).squeeze(0)
    # 2) частотная маска
    f0 = np.random.randint(0, C)
    f_width = np.random.randint(1, max(2, C//10))
    f_width = min(f_width, C - f0)
    clip[f0:f0+f_width] = 0
    # 3) временная маска
    t0 = np.random.randint(0, T)
    t_width = np.random.randint(1, max(2, T//20))
    t_width = min(t_width, T - t0)
    clip[:, t0:t0+t_width] = 0
    return clip

###########################################
# 4. Функции для графа (статическая часть)
###########################################
def compute_adjacency(n_channels):
    coords = np.random.rand(n_channels,2)
    dist = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=2)
    adj = 1.0/(dist + 1e-5)
    np.fill_diagonal(adj, 0)
    return adj / adj.max()

def create_edge_index(adj, threshold=0.1):
    src, dst = np.where((adj > threshold) & (np.eye(adj.shape[0]) == 0))
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)

###########################################
# 1. Dataset + sampler weights
###########################################
class EEGDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.paths, self.labels = [], []
        for subj in os.listdir(root_dir):
            p = os.path.join(root_dir, subj)
            if os.path.isdir(p):
                for f in os.listdir(p):
                    if f.endswith('.npz'):
                        self.paths.append(os.path.join(p, f))
                        self.labels.append(int(np.load(os.path.join(p,f))['label']))
        counts = np.bincount(self.labels)
        weights = 1.0 / counts
        self.sample_weights = [weights[l] for l in self.labels]
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        clip = torch.tensor(data['clip'], dtype=torch.float32)  # [C, T]
        label = int(data['label'])
        if self.augment:
            clip = augment_clip(clip)
        return clip, label

###########################################
# 2. Focal Loss
###########################################
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

###########################################
# 3. SE-блок
###########################################
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels//reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

###########################################
# 4. Multi-scale 1D Conv
###########################################
class MultiScaleConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=[15,25,35]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=k//2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ELU()
            ) for k in kernels
        ])

    def forward(self, x):
        # x: [B*C,1,T]
        return sum(conv(x) for conv in self.convs) / len(self.convs)

###########################################
# 5. Улучшенная модель
###########################################
class RobustHybridEEGModelV2(nn.Module):
    def __init__(self,
                 n_channels=19, n_samples=500,
                 feat_dim=16, gcn_hidden=64, heads=8,
                 num_classes=2, noise_sigma=0.1, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.noise_sigma = noise_sigma

        # 2D-ветвь
        self.conv1    = nn.Conv2d(1, 16, (1,25), padding=(0,12), bias=False)
        self.bn1      = nn.BatchNorm2d(16)
        self.depth    = nn.Conv2d(16,16,(n_channels,1),groups=16,bias=False)
        self.bn2      = nn.BatchNorm2d(16)
        self.elu1     = nn.ELU()
        self.se       = SEBlock(16)
        self.dropb    = DropBlock2D(block_size=3, drop_prob=0.2)
        self.pool1    = nn.AvgPool2d((1,4))
        self.dropout1 = nn.Dropout(dropout)
        self.sep      = nn.Conv2d(16,16,(1,16),padding=(0,8),bias=False)
        self.bn3      = nn.BatchNorm2d(16)
        self.elu2     = nn.ELU()
        self.pool2    = nn.AdaptiveAvgPool2d((self.n_channels,1))

        # 1D-ветвь
        self.mscale   = MultiScaleConv1d(1, feat_dim)
        self.pool1d   = nn.AdaptiveAvgPool1d(1)
        self.dropout2 = nn.Dropout(dropout)

        # динамическая adjacency
        self.adj_lin  = nn.Linear(feat_dim+16, 1)

        # графовый блок
        self.gat1     = GATConv(feat_dim+16, gcn_hidden, heads=heads, concat=False)
        self.bn_g1    = nn.BatchNorm1d(gcn_hidden)
        self.skip1    = nn.Linear(feat_dim+16, gcn_hidden)
        self.gat2     = GATConv(gcn_hidden, gcn_hidden, heads=heads, concat=False)
        self.bn_g2    = nn.BatchNorm1d(gcn_hidden)

        # классификатор
        self.classifier = nn.Sequential(
            nn.Linear(gcn_hidden, gcn_hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(gcn_hidden, num_classes)
        )

    def forward(self, x, edge_index, batch_idx):
        # x: [B,C,T]
        if self.training and self.noise_sigma>0:
            x = x + torch.randn_like(x) * self.noise_sigma
        B,C,T = x.shape

        # 2D ветвь
        x2d = x.unsqueeze(1)
        y = self.elu1(self.bn2(self.depth(self.bn1(self.conv1(x2d)))))
        y = self.se(y)
        y = self.dropb(y)
        y = self.dropout1(self.pool1(y))
        y = self.elu2(self.bn3(self.sep(y)))
        y = self.pool2(y)
        y = y.squeeze(-1).permute(0,2,1)

        # 1D ветвь
        z = x.view(B*C,1,T)
        z = self.mscale(z)
        z = self.pool1d(z).squeeze(-1)
        z = self.dropout2(z)
        z = z.view(B,C,-1)

        # фьюжн
        feat = torch.cat([y,z], dim=2).view(B*C,-1)

        # динамические веса ребер
        scores = torch.sigmoid(self.adj_lin(feat)).view(-1)
        ew = scores[edge_index[0]] * scores[edge_index[1]]

        # графовый блок
        h0 = feat
        h  = self.gat1(feat, edge_index, ew)
        h  = F.gelu(self.bn_g1(h) + self.skip1(h0))
        h  = F.dropout(h, p=0.5, training=self.training)
        h  = self.gat2(h, edge_index, ew)
        h  = F.gelu(self.bn_g2(h) + h)
        h  = F.dropout(h, p=0.5, training=self.training)
        g  = global_mean_pool(h, batch_idx)

        return self.classifier(g)

###########################################
# 6. Цикл обучения с OneCycleLR и ранним стопом
###########################################
def train_and_evaluate(root_dir, device, num_epochs=50):
    ds = EEGDataset(root_dir, augment=True)
    n   = len(ds)
    n_train = int(0.8*n)
    train_ds, val_ds = random_split(ds, [n_train, n-n_train])

    train_weights = [ds.sample_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

    adj        = compute_adjacency(ds[0][0].shape[0])
    edge_index = create_edge_index(adj, threshold=0.3).to(device)

    model     = RobustHybridEEGModelV2().to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    steps     = num_epochs * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=1e-2, total_steps=steps,
                           pct_start=0.1, anneal_strategy='cos')

    best_val, wait = 0.0, 0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    for epoch in range(1, num_epochs+1):
        # train
        model.train()
        tl, tc, tt = 0,0,0
        for x, y in train_loader:
            x,y = x.to(device), y.to(device)
            B,C,T = x.shape
            bi = torch.arange(B,device=device).unsqueeze(1).repeat(1,C).view(-1)
            preds = model(x, edge_index, bi)
            loss  = criterion(preds, y)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            tl += loss.item()*B
            tc += (preds.argmax(1)==y).sum().item()
            tt += B

        history['train_loss'].append(tl/tt)
        history['train_acc'].append(tc/tt)

        # val
        model.eval()
        vl, vc, vt = 0,0,0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                B,C,T = x.shape
                bi = torch.arange(B,device=device).unsqueeze(1).repeat(1,C).view(-1)
                preds = model(x, edge_index, bi)
                loss  = criterion(preds, y)
                vl += loss.item()*B
                vc += (preds.argmax(1)==y).sum().item()
                vt += B

        val_loss = vl/vt; val_acc = vc/vt
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}: train_loss={history['train_loss'][-1]:.4f}, "
              f"train_acc={history['train_acc'][-1]:.4f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val, wait = val_acc, 0
            torch.save(model.state_dict(), "best_model_v2.pth")
        else:
            wait += 1
            if wait >= 10:
                print(f"Early stopping at epoch {epoch}, best val_acc={best_val:.4f}")
                break

    # plot
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'],   label='Val Acc')
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_and_evaluate(
        "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips",
        device
    )
