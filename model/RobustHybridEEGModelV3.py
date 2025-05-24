import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import mne
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.nn import GATConv, global_mean_pool
from dropblock import DropBlock2D  # pip install dropblock
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------
# Параметры
# ---------------------------------------------
SFREQ = 200
SUBCLIP_LEN = 60 * SFREQ
P_CHANNEL_DROPOUT = 0.1
NOISE_SIGMA_AUG = 0.01
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}
CONFIG = {
    'lr': 5e-4,
    'weight_decay': 1e-5,
    'batch_size': 64,
    'epochs': 150,
    'patience': 20,
    'dropout': 0.4
}

# ---------------------------------------------
# 1. Расчёт относительной мощности полос
# ---------------------------------------------
def compute_bandpower_np(clip: np.ndarray, sfreq: int = SFREQ) -> np.ndarray:
    C, T = clip.shape
    freqs = np.fft.rfftfreq(T, 1/sfreq)
    psd = np.abs(np.fft.rfft(clip, axis=1))**2
    total = psd.sum(axis=1, keepdims=True)
    bands = []
    for low, high in BANDS.values():
        idx = (freqs >= low) & (freqs <= high)
        power = psd[:, idx].sum(axis=1, keepdims=True)
        bands.append(power)
    bp = np.concatenate(bands, axis=1) / (total + 1e-12)
    return bp

# ---------------------------------------------
# 2. Аугментации и клipping
# ---------------------------------------------
def augment_clip(clip: torch.Tensor) -> torch.Tensor:
    C, T = clip.shape
    if T > SUBCLIP_LEN:
        start = np.random.randint(0, T - SUBCLIP_LEN + 1)
        clip = clip[:, start:start+SUBCLIP_LEN]
    else:
        clip = F.pad(clip, (0, SUBCLIP_LEN - T))
    mask = torch.rand(C) < P_CHANNEL_DROPOUT
    clip[mask] = 0
    clip += torch.randn_like(clip) * NOISE_SIGMA_AUG
    f0 = np.random.randint(0, C)
    fw = np.random.randint(1, max(2, C//10))
    clip[f0:f0+min(fw, C-f0)] = 0
    t0 = np.random.randint(0, SUBCLIP_LEN)
    tw = np.random.randint(1, max(2, SUBCLIP_LEN//20))
    clip[:, t0:t0+min(tw, SUBCLIP_LEN-t0)] = 0
    return clip

# ---------------------------------------------
# 3. Графовые утилиты
# ---------------------------------------------
def compute_adjacency(channel_names: list) -> np.ndarray:
    montage = mne.channels.make_standard_montage('standard_1020')
    pos = montage.get_positions()['ch_pos']
    coords = np.array([pos[ch][:2] for ch in channel_names])
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    adj = 1.0 / (dist + 1e-5)
    np.fill_diagonal(adj, 0)
    return adj / adj.max()

def create_edge_index(adj: np.ndarray, thr: float=0.2) -> torch.LongTensor:
    src, dst = np.where((adj > thr) & (np.eye(adj.shape[0])==0))
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)

# ---------------------------------------------
# 4. Dataset
# ---------------------------------------------
class EEGDataset(Dataset):
    def __init__(self, root_dir: str, channel_names: list, augment: bool=False):
        self.paths, self.labels = [], []
        for subj in os.listdir(root_dir):
            sd = os.path.join(root_dir, subj)
            if not os.path.isdir(sd): continue
            for f in os.listdir(sd):
                if f.endswith('.npz'):
                    self.paths.append(os.path.join(sd, f))
                    self.labels.append(int(np.load(os.path.join(sd, f))['label']))
        counts = np.bincount(self.labels)
        weights = 1.0 / counts
        self.sample_weights = [weights[l] for l in self.labels]
        self.channel_names = channel_names
        self.augment = augment

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx])
        clip = arr['clip']  # [C,T]
        bp = compute_bandpower_np(clip)
        clip = torch.tensor(clip, dtype=torch.float32)
        bp   = torch.tensor(bp,   dtype=torch.float32)
        if self.augment:
            clip = augment_clip(clip)
        else:
            C, T = clip.shape
            if T > SUBCLIP_LEN:
                clip = clip[:, :SUBCLIP_LEN]
            else:
                clip = F.pad(clip, (0, SUBCLIP_LEN - T))
        return clip, bp, int(arr['label'])

# ---------------------------------------------
# 5. FocalLoss
# ---------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float=2.0): super().__init__(); self.gamma=gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

# ---------------------------------------------
# 6. Блоки извлечения признаков
# ---------------------------------------------
class MultiScale1DBlock(nn.Module):
    def __init__(self, in_ch=1, out_ch=32, kernels=(50,100,200), dropout=0.4):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=k//2, bias=False),
                nn.BatchNorm1d(out_ch), nn.GELU()
            ) for k in kernels
        ])
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_ch, out_ch//4,1,bias=False), nn.GELU(),
            nn.Conv1d(out_ch//4, out_ch,1,bias=False), nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        y = sum(conv(x) for conv in self.convs) / len(self.convs)
        y = y * self.se(y)
        y = y.mean(-1)
        return self.dropout(y)

class SpatialSpectralBlock(nn.Module):
    def __init__(self, n_ch, out_ch=32, k_time=20, dropout=0.4):
        super().__init__()
        self.conv2d = nn.Conv2d(1, out_ch, (n_ch,k_time), padding=(0,k_time//2), bias=False)
        self.bn = nn.BatchNorm2d(out_ch); self.elu = nn.GELU()
        self.dropb = DropBlock2D(block_size=3, drop_prob=0.2)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch//4,1,bias=False), nn.GELU(),
            nn.Conv2d(out_ch//4, out_ch,1,bias=False), nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d((n_ch,1))
    def forward(self, x):
        y = self.conv2d(x); y = self.bn(y); y = self.elu(y)
        y = self.dropb(y); y = y * self.se(y)
        y = self.pool(y).squeeze(-1).permute(0,2,1)
        return y

# ---------------------------------------------
# 7. Гибридная модель
# ---------------------------------------------
class HybridEEGGraphModel(nn.Module):
    def __init__(self, n_ch, feat_dim=32, spat_dim=32,
                 bp_dim=8, gcn_h=128, heads=4,
                 num_classes=2, dropout=0.4):
        super().__init__()
        self.mblock  = MultiScale1DBlock(1, feat_dim, (50,100,200), dropout)
        self.sblock  = SpatialSpectralBlock(n_ch, spat_dim, 20, dropout)
        self.bp_proj = nn.Linear(len(BANDS), bp_dim)
        total_dim   = feat_dim + spat_dim + bp_dim
        self.ca     = nn.Sequential(
            nn.Linear(total_dim, total_dim//2, bias=False),
            nn.GELU(),
            nn.Linear(total_dim//2, total_dim, bias=False),
            nn.Sigmoid()
        )
        self.gat1   = GATConv(total_dim, gcn_h, heads=heads, concat=False)
        self.gat2   = GATConv(gcn_h, gcn_h, heads=heads, concat=False)
        self.skip1  = nn.Linear(total_dim, gcn_h)
        self.bn1    = nn.BatchNorm1d(gcn_h)
        self.bn2    = nn.BatchNorm1d(gcn_h)
        self.classifier = nn.Sequential(
            nn.Linear(gcn_h, gcn_h//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(gcn_h//2, num_classes)
        )

    def forward(self, x, bp, edge_index, batch_idx):
        B, C, T = x.shape
        z       = self.mblock(x.view(B*C,1,T)).view(B,C,-1)
        y       = self.sblock(x.unsqueeze(1))
        bp_feat = self.bp_proj(bp.view(B*C,-1)).view(B,C,-1)
        feat    = torch.cat([y, z, bp_feat], dim=-1)
        feat    = feat * self.ca(feat)
        feat    = feat.view(B*C, -1)
        h0      = feat
        h1      = self.gat1(feat, edge_index)
        h       = F.elu(self.bn1(h1) + self.skip1(h0))
        h       = F.dropout(h, 0.5, self.training)
        h2      = self.gat2(h, edge_index)
        h       = F.elu(self.bn2(h2) + h)
        h       = F.dropout(h, 0.5, self.training)
        g       = global_mean_pool(h, batch_idx)
        return self.classifier(g)

# ---------------------------------------------
# 8. Train & Evaluate + ROC, CM
# ---------------------------------------------
def train_and_evaluate(train_dir, test_dir, channel_names, device):
    train_ds = EEGDataset(train_dir, channel_names, augment=True)
    test_ds  = EEGDataset(test_dir, channel_names, augment=False)
    sampler  = WeightedRandomSampler(train_ds.sample_weights, len(train_ds), True)
    tr_dl    = DataLoader(train_ds, CONFIG['batch_size'], sampler, drop_last=True)
    te_dl    = DataLoader(test_ds, CONFIG['batch_size'], shuffle=False)

    adj        = compute_adjacency(channel_names)
    edge_index = create_edge_index(adj, 0.2).to(device)

    model      = HybridEEGGraphModel(len(channel_names),
                    feat_dim=32, spat_dim=32, bp_dim=8,
                    gcn_h=128, heads=4,
                    dropout=CONFIG['dropout']).to(device)
    criterion  = FocalLoss(2.0)
    optimizer  = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler  = OneCycleLR(optimizer, max_lr=CONFIG['lr']*10,
                    total_steps=CONFIG['epochs']*len(tr_dl), pct_start=0.1)

    best_acc, wait = 0.0, 0
    for epoch in range(1, CONFIG['epochs']+1):
        model.train(); tl, tc, tt = 0,0,0
        for x, bp, y in tr_dl:
            x, bp, y = x.to(device), bp.to(device), y.to(device)
            B, C, T   = x.shape
            batch_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1,C).view(-1)
            logits    = model(x, bp, edge_index, batch_idx)
            loss      = criterion(logits, y)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            tl += loss.item()*B; preds = logits.argmax(1)
            tc += (preds==y).sum().item(); tt += B
        tr_acc = tc/tt

        model.eval(); vl,vc,vt = 0,0,0
        all_probs, all_trues, all_preds = [], [], []
        with torch.no_grad():
            for x, bp, y in te_dl:
                x, bp, y = x.to(device), bp.to(device), y.to(device)
                B, C, T   = x.shape
                batch_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1,C).view(-1)
                logits    = model(x, bp, edge_index, batch_idx)
                loss      = criterion(logits, y)
                probs     = F.softmax(logits, dim=1)[:,1]
                vl += loss.item()*B
                preds     = logits.argmax(1)
                vc += (preds==y).sum().item(); vt += B
                all_probs.append(probs.cpu())
                all_trues.append(y.cpu())
                all_preds.append(preds.cpu())
        te_acc = vc/vt
        print(f"Epoch {epoch}: Train Acc={tr_acc:.4f} | Test Acc={te_acc:.4f}")
        if te_acc > best_acc:
            best_acc = te_acc; wait = 0
            torch.save(model.state_dict(), 'best_model_91%.pt')
        else:
            wait += 1
            if wait >= CONFIG['patience']:
                print(f"Early stopping @ epoch {epoch}, best={best_acc:.4f}")
                break

    # --- После обучения: ROC и Confusion Matrix ---
    y_true  = torch.cat(all_trues).numpy()
    y_probs = torch.cat(all_probs).numpy()
    y_pred  = torch.cat(all_preds).numpy()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    print('Training and evaluation complete')

if __name__ == '__main__':
    channel_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
                     'T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_and_evaluate('preproc_clips_train', 'preproc_clips_test', channel_names, device)
