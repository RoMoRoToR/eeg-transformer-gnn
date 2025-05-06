import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from torch_geometric.nn import global_mean_pool
from optimized_dstgat_attn_model_final import (
    Optimized_DSTGAT_Attn,
    EEGDataset,
    compute_adjacency,
    create_edge_index,
)

###########################################
# Расширенный класс модели с возвратом attention-коэффициентов
###########################################
class Optimized_DSTGAT_Attn_WithAttn(Optimized_DSTGAT_Attn):
    def forward(self, x, edge_index, return_attention_weights=False):
        """
        Если return_attention_weights=True, возвращает кортеж
        (logits, (edge_index_used, attn_weights))
        """
        B, C, T = x.shape
        # 1) временной экстрактор
        x = x.view(B * C, 1, T)
        feats = self.feature_extractor(x)                    # [B*C, F]
        feats = feats.view(B, C, -1).view(B * C, -1)          # [B*C, F]
        # 2) строим пакетный граф
        new_ei = torch.cat([edge_index + i * C for i in range(B)], dim=1).to(feats.device)
        batch_vec = torch.arange(B, device=feats.device) \
                        .unsqueeze(1).repeat(1, C).view(-1)  # [B*C]
        # 3) первый GATConv
        if return_attention_weights:
            h1, (ei1, attn1) = self.gat1(
                feats, new_ei, return_attention_weights=True
            )
        else:
            h1 = self.gat1(feats, new_ei)
            ei1, attn1 = None, None
        h1 = self.gat1_bn(h1)
        h1 = F.gelu(h1 + self.skip1(feats))
        h1 = self.dropout(h1)
        # 4) второй GATConv (без визуализации)
        h2 = self.gat2(h1, new_ei)
        h2 = self.gat2_bn(h2)
        h2 = F.gelu(h2 + h1)
        h2 = self.dropout(h2)
        # 5) глобальный mean-pool + классификация
        out = global_mean_pool(h2, batch_vec)
        logits = self.fc(out)
        if return_attention_weights:
            # вернём только первую «пачку» ребёр (B=1) для простоты
            E0 = edge_index.size(1)
            return logits, (ei1[:, :E0].cpu(), attn1[:E0].cpu())
        else:
            return logits

###########################################
# Функция визуализации attention-карты
###########################################
def plot_attention_map(edge_idx, attn_weights, n_channels, head=0):
    """
    edge_idx: torch.LongTensor[2, E]
    attn_weights: torch.Tensor[E, heads]
    """
    src = edge_idx[0].numpy()
    dst = edge_idx[1].numpy()
    alphas = attn_weights[:, head].numpy()
    # строим матрицу внимания
    A = np.zeros((n_channels, n_channels))
    for i, j, a in zip(src, dst, alphas):
        A[i, j] = a
    # симметризуем
    M = A + A.T
    plt.figure(figsize=(6, 5))
    sns.heatmap(M, cmap='viridis', xticklabels=range(n_channels),
                yticklabels=range(n_channels))
    plt.title(f'Attention Map (Head {head})')
    plt.xlabel('Channel')
    plt.ylabel('Channel')
    plt.tight_layout()
    plt.show()

###########################################
# Основной скрипт
###########################################
if __name__ == "__main__":
    data_root  = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"
    model_path = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/model/optimized_dstgat_attn_model_final.pth"
    n_channels = 19
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Dataset и граф
    dataset   = EEGDataset(data_root)
    print(f"Клипов в датасете: {len(dataset)}")
    adj       = compute_adjacency(n_channels)
    edge_idx  = create_edge_index(adj, threshold=0.3).to(device)

    # 2) Модель с attention
    model = Optimized_DSTGAT_Attn_WithAttn(
        n_channels=n_channels, eeg_feat_dim=16,
        gcn_channels=32, num_classes=2,
        dropout_rate=0.5, gat_heads=4
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3) Выбираем один клип для объяснения (например, индекс 0)
    clip, label = dataset[0]
    clip = clip.unsqueeze(0).to(device)  # [1, C, T]

    # 4) Пропускаем через модель с возвратом attention
    with torch.no_grad():
        logits, (ei_vis, attn_vis) = model(clip, edge_idx, return_attention_weights=True)

    pred = logits.argmax(dim=1).item()
    prob = F.softmax(logits, dim=1)[0,1].item()
    print(f"Ground truth: {label}, Predicted: {pred}, Prob(AD): {prob:.3f}")

    # 5) Визуализируем карту первой головы
    plot_attention_map(ei_vis, attn_vis, n_channels, head=0)
