import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
from torch.utils.data import Dataset, DataLoader, random_split
import os

class EEGDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: путь к каталогу с предобработанными клипами.
        Структура: root_dir/subjectX/*.npz
        """
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
        data = np.load(self.file_paths[idx])
        clip = data['clip']  # ожидается форма [n_channels, n_samples]
        label = data['label']
        clip = torch.tensor(clip, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return clip, label

###########################################
# 1. Слой стохастического шума
###########################################
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x

###########################################
# 2. 2D-ветвь на базе EEGNet2D_Hybrid
###########################################
class DepthwiseConv2d(nn.Module):
    def __init__(self, in_ch, k, padding=0, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=padding,
                            groups=in_ch, bias=bias)
    def forward(self,x): return self.dw(x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k, padding=0, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=padding,
                            groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
    def forward(self,x): return self.pw(self.dw(x))

class EEGNet2D_Hybrid(nn.Module):
    def __init__(self, n_channels=19, n_samples=500, feat_dim=16, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1,25), padding=(0,12), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.elu = nn.ELU()
        self.depthwise = DepthwiseConv2d(16, k=(1,1))
        self.bn2 = nn.BatchNorm2d(16)
        self.separable = SeparableConv2d(16, 16, k=(1,16), padding=(0,8))
        self.bn3 = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d((n_channels,1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, 1, n_channels, n_samples]
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.elu(self.bn2(self.depthwise(x)))
        x = self.elu(self.bn3(self.separable(x)))
        x = self.pool(x)              # [B,16,n_channels,1]
        x = self.dropout(x).squeeze(-1)  # [B,16,n_channels]
        x = x.permute(0,2,1)          # [B,n_channels,16]
        return x

###########################################
# 3. 1D-ветвь на базе EEGNet1D
###########################################
class DepthwiseConv1d(nn.Module):
    def __init__(self, ch, k, padding=0, bias=False):
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, kernel_size=k, padding=padding,
                            groups=ch, bias=bias)
    def forward(self,x): return self.dw(x)

class SeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, padding=0, bias=False):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=padding,
                            groups=in_ch, bias=bias)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)
    def forward(self,x): return self.pw(self.dw(x))

class EEGNet1D(nn.Module):
    def __init__(self, feat_dim=16, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, feat_dim, kernel_size=25, padding=12, bias=False)
        self.bn1 = nn.BatchNorm1d(feat_dim)
        self.elu = nn.ELU()
        self.depthwise = DepthwiseConv1d(feat_dim, k=3, padding=1)
        self.bn2 = nn.BatchNorm1d(feat_dim)
        self.sep = SeparableConv1d(feat_dim, feat_dim, k=3, padding=1)
        self.bn3 = nn.BatchNorm1d(feat_dim)
        self.pool = nn.AvgPool1d(kernel_size=4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B*n_channels,1,n_samples]
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.elu(self.bn2(self.depthwise(x)))
        x = self.elu(self.bn3(self.sep(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.adaptive_avg_pool1d(x,1).squeeze(-1)  # [B*n_channels,feat_dim]
        return x

###########################################
# 4. Функции для графа
###########################################
def compute_adjacency(n_channels):
    coords = np.random.rand(n_channels,2)
    dist = np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=2)
    adj = 1.0/(dist+1e-5)
    np.fill_diagonal(adj,0)
    return adj/adj.max()

def create_edge_index(adj, threshold=0.1):
    src,dst = np.where((adj>threshold)&(np.eye(adj.shape[0])==0))
    return torch.tensor(np.vstack([src,dst]),dtype=torch.long)

###########################################
# 5. Усиленный графовый блок
###########################################
class EnhancedGraphBlock(nn.Module):
    def __init__(self, in_feats, hidden, heads=8, dropout=0.5):
        super().__init__()
        self.gat1 = GATConv(in_feats, hidden, heads=heads, concat=False)
        self.bn1  = nn.BatchNorm1d(hidden)
        self.skip1= nn.Linear(in_feats, hidden)
        self.gat2 = GATConv(hidden, hidden, heads=heads, concat=False)
        self.bn2  = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch_idx):
        # x: [B*n_channels, in_feats]
        x0 = x
        x = self.gat1(x, edge_index)
        x = F.gelu(self.bn1(x) + self.skip1(x0))
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = F.gelu(self.bn2(x) + x)
        x = self.dropout(x)
        # глобальная агрегация
        return global_mean_pool(x, batch_idx)  # [B, hidden]

###########################################
# 6. Финальная модель
###########################################
class RobustHybridEEGModel(nn.Module):
    def __init__(self,
                 n_channels=19,
                 n_samples=500,
                 feat_dim=16,
                 gcn_hidden=64,
                 heads=8,
                 num_classes=2,
                 noise_sigma=0.1,
                 dropout=0.5):
        super().__init__()
        self.noise = GaussianNoise(sigma=noise_sigma)
        self.net2d = EEGNet2D_Hybrid(n_channels, n_samples, feat_dim, dropout)
        self.net1d = EEGNet1D(feat_dim, dropout)
        self.graph = EnhancedGraphBlock(feat_dim*2, gcn_hidden, heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gcn_hidden, gcn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden, num_classes)
        )

    def forward(self, x, edge_index, batch_index):
        # x: [B, n_channels, n_samples]
        x = self.noise(x)
        B, C, T = x.shape

        # 2D-ветвь
        x2d = x.unsqueeze(1)                       # [B,1,C,T]
        f2d = self.net2d(x2d)                      # [B,C,feat_dim]

        # 1D-ветвь
        x1 = x.view(B*C,1,T)                       # [B*C,1,T]
        f1d = self.net1d(x1).view(B, C, -1)        # [B,C,feat_dim]

        # конкатенация признаков
        feats = torch.cat([f2d, f1d], dim=2)       # [B,C,feat_dim*2]
        feats = feats.view(B*C, -1)                # [B*C, feat_dim*2]

        # графовая агрегация
        g = self.graph(feats, edge_index, batch_index)  # [B, gcn_hidden]

        # классификация
        return self.classifier(g)
