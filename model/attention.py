import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from model.RobustHybridEEGModelV3 import (
    EEGDataset, HybridEEGGraphModel,
    compute_adjacency, create_edge_index, CONFIG
)

def plot_attention_weights(edge_index, att, channel_names, title):
    C = len(channel_names)
    # извлекаем numpy-массивы
    src_all, dst_all = edge_index.cpu().numpy()
    alpha = att.cpu().detach().numpy()
    # если многоголовое внимание, усредняем по головам
    if alpha.ndim == 2:
        alpha = alpha.mean(axis=1)
    # отбираем только ребра внутри первого семпла (узлы 0..C-1)
    mask = (src_all < C) & (dst_all < C)
    src = src_all[mask]
    dst = dst_all[mask]
    alpha = alpha[mask]
    # строим матрицу внимания
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



# 1) Параметры
channel_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
                 'T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) Датасет и модель
test_ds     = EEGDataset('preproc_clips_test', channel_names, augment=False)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=True)

model = HybridEEGGraphModel(
    n_ch=len(channel_names),
    feat_dim=32,
    spat_dim=32,
    bp_dim=4,               # ← восстановили исходное bp_dim=4
    gcn_h=128,
    heads=4,
    dropout=CONFIG['dropout']
).to(device)

model.load_state_dict(
    torch.load('best_model.pt', map_location=device)
)
model.eval()

# 3) Берём один батч
x_batch, bp_batch, _ = next(iter(test_loader))
x = x_batch.to(device)
bp= bp_batch.to(device)
B, C, T = x.shape
batch_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1,C).view(-1)

# 4) Строим граф
adj = compute_adjacency(channel_names)
edge_index = create_edge_index(adj, 0.2).to(device)

# 5) Репликация части forward для получения attention_weights
with torch.no_grad():
    # a) извлечение признаков
    # 1D-ветвь
    z = model.mblock(x.view(B*C,1,T)).view(B, C, -1)
    # 2D-ветвь
    y = model.sblock(x.unsqueeze(1))
    # bandpower (если есть)
    bp_feat = model.bp_proj(bp.view(B*C, -1)).view(B, C, -1)
    # фьюжн + channel attention
    feat = torch.cat([y, z, bp_feat], dim=-1)
    feat = feat * model.ca(feat)
    feat = feat.view(B*C, -1)

    # первый GAT с возвратом весов
    h0 = feat
    h1, (edge1, att1) = model.gat1(
        feat, edge_index, return_attention_weights=True
    )
    h = F.elu(model.bn1(h1) + model.skip1(h0))
    h = F.dropout(h, 0.5, training=False)

    # второй GAT с возвратом весов
    h2, (edge2, att2) = model.gat2(
        h, edge_index, return_attention_weights=True
    )

# 6) Визуализируем
plot_attention_weights(edge1, att1, channel_names, 'GAT Layer 1 Attention')
plot_attention_weights(edge2, att2, channel_names, 'GAT Layer 2 Attention')
