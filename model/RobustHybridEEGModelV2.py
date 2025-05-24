import os

import mne
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

# Функция для построения attention-карты (вынесена в глобальную область)
def plot_attention_weights(edge_index, att, channel_names, title):
    C = len(channel_names)
    src_all, dst_all = edge_index.cpu().numpy()
    alpha = att.cpu().detach().numpy()
    if alpha.ndim == 2:
        alpha = alpha.mean(axis=1)
    mask = (src_all < C) & (dst_all < C)
    src = src_all[mask]
    dst = dst_all[mask]
    alpha = alpha[mask]
    mat = np.zeros((C, C))
    for i, j, w in zip(src, dst, alpha):
        mat[i, j] = float(w)
    plt.figure(figsize=(6,5))
    plt.imshow(mat, cmap='viridis', vmin=0, vmax=alpha.max())
    plt.colorbar(label='Attention weight')
    plt.xticks(range(C), channel_names, rotation=90)
    plt.yticks(range(C), channel_names)
    plt.title(title)
    plt.tight_layout()
    plt.show()

###########################################
# Поведенческие аугментации
###########################################
def augment_clip(clip):
    C, T = clip.shape
    crop = int(0.9 * T)
    start = np.random.randint(0, T - crop + 1)
    clip = clip[:, start:start+crop]
    clip = F.interpolate(clip.unsqueeze(0), size=T, mode='linear', align_corners=False).squeeze(0)
    f0 = np.random.randint(0, C)
    f_width = np.random.randint(1, max(2, C//10))
    f_width = min(f_width, C - f0)
    clip[f0:f0+f_width] = 0
    t0 = np.random.randint(0, T)
    t_width = np.random.randint(1, max(2, T//20))
    t_width = min(t_width, T - t0)
    clip[:, t0:t0+t_width] = 0
    return clip

###########################################
# Графовые утилиты
###########################################
def compute_adjacency(channel_names: list) -> np.ndarray:
    montage = mne.channels.make_standard_montage('standard_1020')
    pos = montage.get_positions()['ch_pos']
    coords = np.array([pos[ch][:2] for ch in channel_names])
    # стандартизация координат: zero-mean, unit-variance
    coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    adj = 1.0 / (dist + 1e-5)
    np.fill_diagonal(adj, 0)
    return adj / adj.max()

def create_edge_index(adj, threshold=0.1):
    src, dst = np.where((adj > threshold) & (np.eye(adj.shape[0]) == 0))
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)

###########################################
# Dataset + sampler weights
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
                        self.labels.append(int(np.load(os.path.join(p, f))['label']))
        counts = np.bincount(self.labels)
        weights = 1.0 / counts
        self.sample_weights = [weights[l] for l in self.labels]
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        clip = torch.tensor(data['clip'], dtype=torch.float32)
        label = int(data['label'])
        if self.augment:
            clip = augment_clip(clip)
        return clip, label

###########################################
# Focal Loss
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
# SE-блок
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
# Multi-scale 1D Conv
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
        return sum(conv(x) for conv in self.convs) / len(self.convs)

###########################################
# Модель с визуализацией attention
###########################################
class RobustHybridEEGModelV2(nn.Module):
    def __init__(self, n_channels=19, feat_dim=16, gcn_hidden=64, heads=8, num_classes=2, noise_sigma=0.1, dropout=0.5):
        super().__init__()
        self.noise_sigma = noise_sigma
        # 2D-ветвь
        self.conv1 = nn.Conv2d(1, 16, (1,25), padding=(0,12), bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.depth = nn.Conv2d(16,16,(n_channels,1),groups=16,bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.elu1  = nn.ELU()
        self.se    = SEBlock(16)
        self.dropb = DropBlock2D(block_size=3, drop_prob=0.2)
        self.pool1 = nn.AvgPool2d((1,4))
        self.dp1   = nn.Dropout(dropout)
        self.sep   = nn.Conv2d(16,16,(1,16),padding=(0,8),bias=False)
        self.bn3   = nn.BatchNorm2d(16)
        self.elu2  = nn.ELU()
        self.pool2 = nn.AdaptiveAvgPool2d((n_channels,1))
        # 1D-ветвь
        self.mscale   = MultiScaleConv1d(1, feat_dim)
        self.pool1d   = nn.AdaptiveAvgPool1d(1)
        self.dp2      = nn.Dropout(dropout)
        self.adj_lin  = nn.Linear(feat_dim+16, 1)
        # GAT
        self.gat1   = GATConv(feat_dim+16, gcn_hidden, heads=heads, concat=False)
        self.bn_g1  = nn.BatchNorm1d(gcn_hidden)
        self.skip1  = nn.Linear(feat_dim+16, gcn_hidden)
        self.gat2   = GATConv(gcn_hidden, gcn_hidden, heads=heads, concat=False)
        self.bn_g2  = nn.BatchNorm1d(gcn_hidden)
        # Classifier
        self.classifier = nn.Sequential(nn.Linear(gcn_hidden,gcn_hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(gcn_hidden,num_classes))
        # Для attention
        self.attn_edge_index = None
        self.attn_weights    = None
    def forward(self, x, edge_index, batch_idx):
        if self.training and self.noise_sigma>0:
            x = x + torch.randn_like(x)*self.noise_sigma
        B,C,T = x.shape
        # 2D-ветвь
        y = self.elu1(self.bn2(self.depth(self.bn1(self.conv1(x.unsqueeze(1))))))
        y = self.se(y); y = self.dropb(y); y = self.dp1(self.pool1(y))
        y = self.elu2(self.bn3(self.sep(y)))
        y = self.pool2(y)
        y = y.squeeze(-1).permute(0,2,1)
        # 1D-ветвь
        z = self.mscale(x.view(B*C,1,T))
        z = self.pool1d(z).squeeze(-1); z = self.dp2(z)
        z = z.view(B,C,-1)
        feat = torch.cat([y,z],dim=2).view(B*C,-1)
        # динамические веса (опционально)
        scores = torch.sigmoid(self.adj_lin(feat)).view(-1)
        # GATConv с возвратом attention
        h0 = feat
        h1, (edge_idx, att_w) = self.gat1(feat, edge_index, return_attention_weights=True)
        self.attn_edge_index = edge_idx
        self.attn_weights    = att_w
        h = F.gelu(self.bn_g1(h1) + self.skip1(h0)); h = F.dropout(h,0.5,self.training)
        h2 = self.gat2(h, edge_index)
        h = F.gelu(self.bn_g2(h2) + h); h = F.dropout(h,0.5,self.training)
        g = global_mean_pool(h,batch_idx)
        return self.classifier(g)

###########################################
# Train/Eval + визуализация attention
###########################################
def train_and_evaluate(root_dir, channel_names, device, num_epochs=50):
    ds = EEGDataset(root_dir, augment=True)
    n_train = int(0.8*len(ds))
    train_ds, val_ds = random_split(ds,[n_train,len(ds)-n_train])
    sampler = WeightedRandomSampler([ds.sample_weights[i] for i in train_ds.indices],len(train_ds),True)
    tr_dl = DataLoader(train_ds,batch_size=16,sampler=sampler,drop_last=True)
    val_dl= DataLoader(val_ds,batch_size=16,shuffle=False)
    adj = compute_adjacency(channel_names); edge_index = create_edge_index(adj,0.3).to(device)
    model = RobustHybridEEGModelV2().to(device)
    opt   = optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    sched = OneCycleLR(opt,max_lr=1e-2,total_steps=num_epochs*len(tr_dl),pct_start=0.1)
    crit = FocalLoss(2.0)
    best,wait=0,0; history = {'tl':[],'ta':[],'vl':[],'va':[]}
    for epoch in range(1,num_epochs+1):
        model.train(); tl,tc,tt=0,0,0
        for x,y in tr_dl:
            x,y=x.to(device),y.to(device)
            bi=torch.arange(x.size(0),device=device).unsqueeze(1).repeat(1,x.size(1)).view(-1)
            preds= model(x,edge_index,bi); loss=crit(preds,y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step()
            tl+=loss.item()*x.size(0); tc+=(preds.argmax(1)==y).sum().item(); tt+=x.size(0)
        history['tl'].append(tl/tt); history['ta'].append(tc/tt)
        model.eval(); vl,vc,vt=0,0,0
        with torch.no_grad():
            for x,y in val_dl:
                x,y=x.to(device),y.to(device)
                bi=torch.arange(x.size(0),device=device).unsqueeze(1).repeat(1,x.size(1)).view(-1)
                preds = model(x,edge_index,bi); loss=crit(preds,y)
                vl+=loss.item()*x.size(0); vc+=(preds.argmax(1)==y).sum().item(); vt+=x.size(0)
        va=vc/vt
        history['vl'].append(vl/vt); history['va'].append(va)
        print(f"Epoch {epoch}: train_acc={history['ta'][-1]:.4f} | val_acc={va:.4f}")
        if va>best: best,wait=va,0; torch.save(model.state_dict(),"best.pth")
        else: wait+=1
        if wait>=10: print("Early stopping"); break
    # plot metrics
    ep=range(1,len(history['tl'])+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(ep, history['tl'],label='Train Loss'); plt.plot(ep, history['vl'],label='Val Loss'); plt.legend()
    plt.subplot(1,2,2); plt.plot(ep, history['ta'],label='Train Acc'); plt.plot(ep, history['va'],label='Val Acc'); plt.legend(); plt.show()
    # visualize attention using plot_attention_weights
    # получаем одну партию из валидационного даталоадера
    x_val, _ = next(iter(val_dl))
    x_val = x_val.to(device)
    bi = torch.arange(x_val.size(0), device=device).unsqueeze(1).repeat(1, x_val.size(1)).view(-1)
    _ = model(x_val, edge_index, bi)
    edge_idx, att_w = model.attn_edge_index, model.attn_weights
    # строим карту внимания
    plot_attention_weights(edge_idx, att_w, channel_names, 'GAT Layer1 Attention')

if __name__ == "__main__":
    # 1) Параметры
    channel_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
                     'T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Тренировка и оценка
    train_and_evaluate(
        "preproc_clips",
        channel_names,
        device
    )

    # ------------------------------
    # 3) Визуализация attention на тесте
    # ------------------------------
    from torch.utils.data import DataLoader

    # Функция для построения attention-карты
    def plot_attention_weights(edge_index, att, channel_names, title):
        C = len(channel_names)
        src_all, dst_all = edge_index.cpu().numpy()
        alpha = att.cpu().detach().numpy()
        if alpha.ndim == 2:
            alpha = alpha.mean(axis=1)
        mask = (src_all < C) & (dst_all < C)
        src = src_all[mask]
        dst = dst_all[mask]
        alpha = alpha[mask]
        mat = np.zeros((C, C))
        for i, j, w in zip(src, dst, alpha):
            mat[i, j] = float(w)
        plt.figure(figsize=(6,5))
        plt.imshow(mat, cmap='viridis', vmin=0, vmax=alpha.max())
        plt.colorbar(label='Attention weight')
        plt.xticks(range(C), channel_names, rotation=90)
        plt.yticks(range(C), channel_names)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # Загружаем датасет и модель
    test_ds = EEGDataset('preproc_clips_test', augment=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=True)

    # Загружаем обученную модель
    model = RobustHybridEEGModelV2(
        n_channels=len(channel_names)
    ).to(device)
    model.load_state_dict(torch.load('best_model_v2.pth', map_location=device))
    model.eval()

    # Берем один батч, повторяем часть forward для gat1 и gat2
    x_batch, y_batch = next(iter(test_loader))
    x = x_batch.to(device)
    B, C, T = x.shape
    batch_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1, C).view(-1)

    # Собираем граф
    adj = compute_adjacency(channel_names)
    edge_index = create_edge_index(adj, threshold=0.3).to(device)

    with torch.no_grad():
        # 1D и 2D ветви
        z = model.mscale(x.view(B*C,1,T))
        z = model.pool1d(z).squeeze(-1).view(B, C, -1)
        y = model.conv1(x.unsqueeze(1))
        y = model.bn1(y); y = model.depth(y); y = model.bn2(y); y = model.elu1(y)
        y = model.se(y); y = model.dropb(y); y = model.dp1(model.pool1(y))
        y = model.elu2(model.bn3(model.sep(y)))
        y = model.pool2(y).squeeze(-1).permute(0,2,1)
        # Fusion
        feat = torch.cat([y, z], dim=2).view(B*C, -1)
        # GAT1
        h0 = feat
        h1, (edge1, att1) = model.gat1(feat, edge_index, return_attention_weights=True)
        h = F.elu(model.bn_g1(h1) + model.skip1(h0))
        h = F.dropout(h, 0.5, training=False)
        # GAT2
        h2, (edge2, att2) = model.gat2(h, edge_index, return_attention_weights=True)

    # 4) Рисуем
    plot_attention_weights(edge1, att1, channel_names, 'GAT Layer 1 Attention')
    plot_attention_weights(edge2, att2, channel_names, 'GAT Layer 2 Attention')
