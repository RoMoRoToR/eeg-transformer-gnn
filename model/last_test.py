#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobustHybridEEGModelV2 + стандартный 10-20 монтаж
=================================================

• графовая adjacency формируется по реальным координатам электродов
  (Fp1, Fp2, … O2) из MNE-montage 'standard_1020';
• веса рёбер = 1/(distance+ε), далее двоичная маска порогом >thr;
• всё остальное (аугментации, focal loss, OneCycleLR, DropBlock2D)
  оставлено без изменений.
"""

import os, sys, math, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.nn import GATConv, global_mean_pool
from dropblock import DropBlock2D            # pip install dropblock
import mne                                   # координаты монтажа

# ─────────────────────────────────────────────────────────────────────────────
# 0. ПАРАМЕТРЫ ДАТАСЕТА И МОНТАЖА
# ─────────────────────────────────────────────────────────────────────────────
CHANNEL_NAMES = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'
]

def build_adjacency(channel_names, threshold=0.1):
    """
    channel_names : list(str) в том же порядке, что и в клипах [C,T]
    threshold     : порог бинаризации на нормированной матрице (0..1)

    Возвращает:
        adj_norm  : ndarray [C,C] (значения 0..1)
        edge_index: LongTensor [2, E]
    """
    # — координаты xyz из стандартного 10-20 (метры, RAS) —
    montage = mne.channels.make_standard_montage('standard_1020')
    pos = montage.get_positions()['ch_pos']

    coords = np.array([pos[ch] for ch in channel_names])     # [C,3]
    # — попарное euclidean расстояние —
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    # — обращаем и нормируем до 0..1 —
    with np.errstate(divide='ignore'):
        adj = 1.0 / (dists + 1e-12)
    np.fill_diagonal(adj, 0.0)
    adj /= adj.max()

    # — edge_index по порогу —
    src, dst = np.where((adj > threshold) & (adj > 0))
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    return adj, edge_index


# ─────────────────────────────────────────────────────────────────────────────
# 1. АУГМЕНТАЦИИ ДЛЯ КЛИПОВ
# ─────────────────────────────────────────────────────────────────────────────
def augment_clip(clip: torch.Tensor) -> torch.Tensor:
    # clip: [C,T]
    C, T = clip.shape
    # 1) время: тайм-кроп 90 %
    crop = int(0.9 * T)
    st   = np.random.randint(0, T - crop + 1)
    clip = clip[:, st:st+crop]
    clip = F.interpolate(clip.unsqueeze(0), size=T,
                         mode='linear', align_corners=False).squeeze(0)
    # 2) частотная маска (каналы)
    f0 = np.random.randint(0, C)
    f_w = np.random.randint(1, max(2, C//10))
    f_w = min(f_w, C - f0)
    clip[f0:f0+f_w] = 0
    # 3) временная маска
    t0 = np.random.randint(0, T)
    t_w = np.random.randint(1, max(2, T//20))
    t_w = min(t_w, T - t0)
    clip[:, t0:t0+t_w] = 0
    return clip


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET (+ ВЕСА ДЛЯ СЭМПЛЕРА)
# ─────────────────────────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, root_dir: str, augment=False):
        self.paths, self.labels = [], []
        for subj in sorted(os.listdir(root_dir)):
            p = os.path.join(root_dir, subj)
            if os.path.isdir(p):
                for f in sorted(os.listdir(p)):
                    if f.endswith('.npz'):
                        self.paths.append(os.path.join(p, f))
                        self.labels.append(int(np.load(os.path.join(p, f))['label']))
        counts = np.bincount(self.labels)
        weights = 1.0 / counts
        self.sample_weights = [weights[l] for l in self.labels]
        self.augment = augment

    def __len__(self):  return len(self.paths)

    def __getitem__(self, idx):
        d = np.load(self.paths[idx])
        clip  = torch.tensor(d['clip'], dtype=torch.float32)   # [C,T]
        label = int(d['label'])
        if self.augment:
            clip = augment_clip(clip)
        return clip, label


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOSS, SE-BLOCK, MULTISCALE 1D
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):  super().__init__(); self.gamma = gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

class SEBlock(nn.Module):
    def __init__(self, ch, red=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch//red, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch//red, ch, 1, bias=False),
            nn.Sigmoid())
    def forward(self, x):  return x * self.net(x)

class MultiScaleConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(15,25,35)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=k//2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ELU()) for k in kernels])
    def forward(self, x):  # [B*C,1,T]
        return sum(conv(x) for conv in self.convs) / len(self.convs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROBUST HYBRID EEG MODEL V2 (не менялся)
# ─────────────────────────────────────────────────────────────────────────────
class RobustHybridEEGModelV2(nn.Module):
    def __init__(self, n_channels=19, n_samples=500,
                 feat_dim=16, gcn_hidden=64, heads=8,
                 num_classes=2, noise_sigma=0.1, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.noise_sigma = noise_sigma
        # 2D-ветвь
        self.conv1 = nn.Conv2d(1,16,(1,25),padding=(0,12),bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.depth = nn.Conv2d(16,16,(n_channels,1),groups=16,bias=False)
        self.bn2   = nn.BatchNorm2d(16); self.elu1 = nn.ELU()
        self.se    = SEBlock(16)
        self.dropb = DropBlock2D(block_size=3, drop_prob=0.2)
        self.pool1 = nn.AvgPool2d((1,4)); self.dropout1 = nn.Dropout(dropout)
        self.sep   = nn.Conv2d(16,16,(1,16),padding=(0,8),bias=False)
        self.bn3   = nn.BatchNorm2d(16); self.elu2 = nn.ELU()
        self.pool2 = nn.AdaptiveAvgPool2d((n_channels,1))
        # 1D-ветвь
        self.mscale  = MultiScaleConv1d(1, feat_dim)
        self.pool1d  = nn.AdaptiveAvgPool1d(1); self.dropout2 = nn.Dropout(dropout)
        # динамическая adjacency
        self.adj_lin = nn.Linear(feat_dim+16, 1)
        # GAT блок
        self.gat1  = GATConv(feat_dim+16, gcn_hidden, heads=heads, concat=False)
        self.bn_g1 = nn.BatchNorm1d(gcn_hidden)
        self.skip1 = nn.Linear(feat_dim+16, gcn_hidden)
        self.gat2  = GATConv(gcn_hidden, gcn_hidden, heads=heads, concat=False)
        self.bn_g2 = nn.BatchNorm1d(gcn_hidden)
        # классификатор
        self.classifier = nn.Sequential(
            nn.Linear(gcn_hidden,gcn_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(gcn_hidden,num_classes))

    def forward(self, x, edge_index, batch_idx):
        # x: [B,C,T]
        if self.training and self.noise_sigma>0:
            x = x + torch.randn_like(x)*self.noise_sigma
        B,C,T = x.shape

        # 2D
        y = x.unsqueeze(1)
        y = self.elu1(self.bn2(self.depth(self.bn1(self.conv1(y)))))
        y = self.se(y); y = self.dropb(y); y = self.dropout1(self.pool1(y))
        y = self.elu2(self.bn3(self.sep(y))); y = self.pool2(y)
        y = y.squeeze(-1).permute(0,2,1)         # [B,C,16]

        # 1D
        z = x.view(B*C,1,T); z = self.mscale(z); z = self.pool1d(z).squeeze(-1)
        z = self.dropout2(z).view(B,C,-1)        # [B,C,feat_dim]

        feat = torch.cat([y,z], dim=2).view(B*C,-1)      # [B*C,F]

        # динамические веса ребер
        scores = torch.sigmoid(self.adj_lin(feat)).view(-1)
        ew = scores[edge_index[0]] * scores[edge_index[1]]

        # GAT
        h0 = feat
        h  = self.gat1(feat, edge_index, ew)
        h  = F.gelu(self.bn_g1(h) + self.skip1(h0))
        h  = F.dropout(h, p=0.5, training=self.training)
        h  = self.gat2(h, edge_index, ew)
        h  = F.gelu(self.bn_g2(h) + h)
        h  = F.dropout(h, p=0.5, training=self.training)

        g  = global_mean_pool(h, batch_idx)      # [B,gcn_hidden]
        return self.classifier(g)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ЦИКЛ ОБУЧЕНИЯ + OneCycleLR + ранний стоп
# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(root_dir, device, num_epochs=50):
    ds = EEGDataset(root_dir, augment=True)
    n  = len(ds); n_train = int(0.8*n)
    train_ds, val_ds = random_split(ds, [n_train, n-n_train],
                                    generator=torch.Generator().manual_seed(42))

    train_weights = [ds.sample_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(train_weights,
                                    num_samples=len(train_weights),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

    # ── ГРАФ (standard_1020) ──────────────────────────────────────────────
    _, edge_index = build_adjacency(CHANNEL_NAMES, threshold=0.3)
    edge_index = edge_index.to(device)

    model = RobustHybridEEGModelV2().to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched     = OneCycleLR(optimizer, max_lr=1e-2,
                           total_steps=num_epochs*len(train_loader),
                           pct_start=0.1, anneal_strategy='cos')

    best_val, wait = 0.0, 0
    hist = {'tr_loss':[], 'tr_acc':[], 'vl_loss':[], 'vl_acc':[]}

    for ep in range(1, num_epochs+1):
        # — TRAIN —
        model.train(); tl, tc, tt = 0,0,0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            B,C,T = x.shape
            batch_idx = torch.arange(B, device=device).unsqueeze(1)\
                         .repeat(1,C).view(-1)
            preds = model(x, edge_index, batch_idx)
            loss  = criterion(preds, y)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); sched.step()
            tl += loss.item()*B
            tc += (preds.argmax(1)==y).sum().item(); tt += B
        hist['tr_loss'].append(tl/tt); hist['tr_acc'].append(tc/tt)

        # — VAL —
        model.eval(); vl, vc, vt = 0,0,0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                B,C,T = x.shape
                batch_idx = torch.arange(B, device=device).unsqueeze(1)\
                             .repeat(1,C).view(-1)
                preds = model(x, edge_index, batch_idx)
                loss  = criterion(preds, y)
                vl += loss.item()*B
                vc += (preds.argmax(1)==y).sum().item(); vt += B
        val_loss, val_acc = vl/vt, vc/vt
        hist['vl_loss'].append(val_loss); hist['vl_acc'].append(val_acc)

        print(f"Epoch {ep:02d}: "
              f"train_loss={hist['tr_loss'][-1]:.4f}, "
              f"train_acc={hist['tr_acc'][-1]:.4f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val, wait = val_acc, 0
            torch.save(model.state_dict(), 'best_model_v2.pth')
        else:
            wait += 1
            if wait >= 10:
                print(f"→ Early stopping (best val_acc={best_val:.4f})")
                break

    # — ГРАФИКИ —
    ep = range(1, len(hist['tr_loss'])+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(ep, hist['tr_loss'], label='train')
    plt.plot(ep, hist['vl_loss'], label='val'); plt.title('Loss'); plt.legend()
    plt.subplot(1,2,2); plt.plot(ep, hist['tr_acc'], label='train')
    plt.plot(ep, hist['vl_acc'], label='val'); plt.title('Accuracy'); plt.legend()
    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
        default='/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips',
        help='Папка с сохранёнными npz-клипами')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_and_evaluate(args.data, device, num_epochs=args.epochs)
