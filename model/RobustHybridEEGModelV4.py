import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mne
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch_geometric.nn import GATConv, global_mean_pool
from dropblock import DropBlock2D  # pip install dropblock

# ----------------------------
# Mixup-аугентация
# ----------------------------
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam

# ----------------------------
# SpecAugment + шум
# ----------------------------
def augment_clip(clip):
    C, T = clip.shape
    crop = int(0.9 * T)
    start = np.random.randint(0, T - crop + 1)
    clip = clip[:, start:start+crop]
    clip = F.interpolate(clip.unsqueeze(0), size=T, mode='linear', align_corners=False).squeeze(0)
    f0 = np.random.randint(0, C); fw = np.random.randint(1, max(2, C//10))
    clip[f0:f0+fw] = 0
    t0 = np.random.randint(0, T); tw = np.random.randint(1, max(2, T//20))
    clip[:, t0:t0+tw] = 0
    clip += torch.randn_like(clip) * 0.01
    return clip

# ----------------------------
# Строим edge_index по 10-20 монтировке
# ----------------------------
def compute_edge_index(n_channels, threshold=0.3):
    montage = mne.channels.make_standard_montage('standard_1020')
    pos = montage.get_positions()['ch_pos']
    chs = list(pos.keys())[:n_channels]
    coords = np.array([pos[ch] for ch in chs])
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    adj = 1.0 / (dist + 1e-5)
    np.fill_diagonal(adj, 0)
    adj /= adj.max()
    src, dst = np.where(adj > threshold)
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)

# ----------------------------
# Датасет
# ----------------------------
class EEGDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx], allow_pickle=True)
        clip = torch.tensor(data['clip'], dtype=torch.float32)
        if self.augment:
            clip = augment_clip(clip)
        label = torch.tensor(int(data['label']), dtype=torch.long)
        return clip, label

# ----------------------------
# Спектральная ветвь (динамический bandpower)
# ----------------------------
class SpectralBranch(nn.Module):
    def __init__(self,
                 bands=[(1,4),(4,8),(8,13),(13,30),(30,45)],
                 feat_dim=16,
                 sfreq=200):
        super().__init__()
        self.bands = bands
        self.sfreq = sfreq
        self.fc    = nn.Linear(len(bands), feat_dim)

    def forward(self, x):
        B, C, T = x.shape
        freqs = torch.fft.rfftfreq(T, d=1.0/self.sfreq).to(x.device)
        X = torch.fft.rfft(x, dim=2)
        psd = (X.abs()**2) / T  # (B, C, F)
        outs = []
        for f1, f2 in self.bands:
            mask = (freqs >= f1) & (freqs < f2)
            outs.append(psd[:,:,mask].mean(dim=2))  # (B, C)
        s = torch.stack(outs, dim=2)               # (B, C, n_bands)
        return F.elu(self.fc(s))                   # (B, C, feat_dim)

# ----------------------------
# Модель EEGNet + GAT
# ----------------------------
class EEGNetGAT(nn.Module):
    def __init__(self,
                 n_channels=19,
                 depth_k=25,
                 temp_filt=16,
                 sep_filt=16,
                 spec_dim=16,
                 gcn_dim=64,
                 heads=4,
                 num_classes=2):
        super().__init__()
        self.nC = n_channels

        # EEGNet-ветвь
        self.conv1     = nn.Conv2d(1, temp_filt, (1, depth_k),
                                   padding=(0, depth_k//2), bias=False)
        self.bn1       = nn.BatchNorm2d(temp_filt)
        self.separable = nn.Conv2d(temp_filt, sep_filt,
                                   (1,16), padding=(0,8), bias=False)
        self.bn3       = nn.BatchNorm2d(sep_filt)
        self.pool      = nn.AvgPool2d((1,4))
        self.dropb     = DropBlock2D(block_size=3, drop_prob=0.2)

        # Спектральная ветвь
        self.spec      = SpectralBranch(sfreq=200, feat_dim=spec_dim)

        # Fusion + динамические веса ребёр
        fusion         = sep_filt + spec_dim
        self.adj_lin   = nn.Linear(fusion, 1)

        # GAT-слои
        self.gat1      = GATConv(fusion, gcn_dim, heads=heads, concat=False)
        self.bn_g1     = nn.BatchNorm1d(gcn_dim)
        self.gat2      = GATConv(gcn_dim, gcn_dim, heads=heads, concat=False)
        self.bn_g2     = nn.BatchNorm1d(gcn_dim)
        self.skip      = nn.Linear(fusion, gcn_dim)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(gcn_dim, gcn_dim), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(gcn_dim, num_classes)
        )

    def forward(self, x, edge_index, batch_idx):
        B, C, T = x.shape

        # --- EEGNet-ветвь ---
        y = self.conv1(x.unsqueeze(1))      # (B, temp_filt, C, T)
        y = F.elu(self.bn1(y))
        y = self.separable(y)               # (B, sep_filt, C, T)
        y = F.elu(self.bn3(y))
        y = self.pool(y)                    # (B, sep_filt, C, T//4)
        y = y.mean(dim=3)                   # (B, sep_filt, C)
        y = y.permute(0,2,1)                # (B, C, sep_filt)
        y = self.dropb(y.unsqueeze(-1)).squeeze(-1)  # (B, C, sep_filt)

        # --- Спектральная ветвь ---
        s = self.spec(x)                    # (B, C, spec_dim)

        # --- Fusion & GAT ---
        feat = torch.cat([y, s], dim=2)     # (B, C, fusion)
        feat = feat.view(B*C, -1)           # (B*C, fusion)
        w    = torch.sigmoid(self.adj_lin(feat)).view(-1)
        ew   = w[edge_index[0]] * w[edge_index[1]]

        h0   = feat
        h    = F.elu(self.bn_g1(self.gat1(feat, edge_index, ew)))
        h    = F.dropout(h + self.skip(h0), 0.5, self.training)
        h2   = F.elu(self.bn_g2(self.gat2(h, edge_index, ew)))

        # --- Readout & classify ---
        g    = global_mean_pool(h2, batch_idx)  # (B, gcn_dim)
        return self.classifier(g)

# ----------------------------
# Цикл обучения
# ----------------------------
def train_on_dirs(train_root, test_root, device, epochs=100):
    def collect(root):
        ps, ls = [], []
        for subj in os.listdir(root):
            sd = os.path.join(root, subj)
            if not os.path.isdir(sd): continue
            for f in sorted(os.listdir(sd)):
                if f.endswith('.npz'):
                    path = os.path.join(sd, f)
                    ps.append(path)
                    ls.append(int(np.load(path, allow_pickle=True)['label']))
        return ps, ls

    train_ps, train_ls = collect(train_root)
    val_ps,   val_ls   = collect(test_root)

    # сбалансированный семплер
    weights = 1.0 / np.bincount(train_ls)
    sampler = WeightedRandomSampler([weights[l] for l in train_ls],
                                    len(train_ps), replacement=True)

    train_ds = EEGDataset(train_ps, train_ls, augment=True)
    val_ds   = EEGDataset(val_ps,   val_ls,   augment=False)
    tr = DataLoader(train_ds, batch_size=32, sampler=sampler, drop_last=True)
    va = DataLoader(val_ds,   batch_size=32, shuffle=False)

    ei = compute_edge_index(train_ds[0][0].shape[0]).to(device)
    model = EEGNetGAT().to(device)
    opt   = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = SequentialLR(opt,
        [LinearLR(opt, start_factor=0.2, end_factor=1.0, total_iters=5),
         CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1)],
        milestones=[5])
    swa_m  = AveragedModel(model)
    swa_st = int(0.5 * epochs)
    swa_s  = SWALR(opt, swa_lr=1e-3)

    best, wait = 0.0, 0
    alpha, smooth = 0.2, 0.1

    for ep in range(1, epochs+1):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            xm, ya, yb, lam = mixup_data(x, y, alpha)
            B, C, T = xm.shape
            bi = torch.arange(B, device=device).unsqueeze(1).repeat(1, C).view(-1)

            preds = model(xm, ei, bi)
            loss  = lam * F.cross_entropy(preds, ya, label_smoothing=smooth) + \
                    (1-lam) * F.cross_entropy(preds, yb, label_smoothing=smooth)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if ep > swa_st:
            swa_m.update_parameters(model); swa_s.step()

        model.eval()
        corr = tot = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                B, C, T = x.shape
                bi = torch.arange(B, device=device).unsqueeze(1).repeat(1, C).view(-1)
                corr += (model(x, ei, bi).argmax(1) == y).sum().item()
                tot  += B
        acc = corr / tot
        print(f"Epoch {ep}: Val Acc {acc:.4f}")
        if acc > best:
            best, wait = acc, 0
        else:
            wait += 1
            if wait >= 10:
                print("Early stopping"); break

    # финализация SWA: обновляем BN-статистики
    def swa_fwd(x):
        B, C, T = x.shape
        bi = torch.arange(B, device=device).unsqueeze(1).repeat(1, C).view(-1)
        return swa_m.module(x, ei, bi)
    swa_m.forward = swa_fwd

    bnldr = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: torch.stack([i[0] for i in b]).to(device)
    )
    update_bn(bnldr, swa_m)
    torch.save(swa_m.state_dict(), "eegnet_gat_swa.pth")

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", dev)
    train_on_dirs("preproc_clips_train", "preproc_clips_test", dev)
