import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from torch_geometric.nn import global_mean_pool
from improved_dstgat_attn_model import Improved_DSTGAT_Attn, EEGDataset, compute_adjacency, create_edge_index  # ваш файл с моделью

def test_with_noise(model, dataset, edge_index, device, noise_levels):
    accs = []
    aucs = []
    for sigma in noise_levels:
        preds, labels, probs = [], [], []
        model.eval()
        with torch.no_grad():
            for clip, label in dataset:
                # добавляем шум
                noisy = clip + sigma * torch.randn_like(clip)
                noisy = noisy.unsqueeze(0).to(device)  # [1, n_chan, n_samples]
                out = model(noisy, edge_index)
                prob = F.softmax(out, dim=1)[0,1].item()
                _, pred = out.max(1)
                preds.append(pred.item())
                labels.append(label.item())
                probs.append(prob)
        # метрики
        correct = sum(p==l for p,l in zip(preds,labels))
        acc = correct/len(labels)*100
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr,tpr)*100
        accs.append(acc)
        aucs.append(roc_auc)
        print(f"σ={sigma:.2f} → Acc={acc:.1f}%, AUC={roc_auc:.1f}%")
    return accs, aucs

if __name__ == "__main__":
    # 1) Параметры
    data_root  = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"
    model_path = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/model/improved_dstgcn_model_final.pth"
    n_channels = 19
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Загрузка датасета и модели
    dataset    = EEGDataset(data_root)
    adj        = compute_adjacency(n_channels)
    edge_index = create_edge_index(adj, threshold=0.3).to(device)

    model = Improved_DSTGAT_Attn(
        n_channels=n_channels,
        kernel_sizes=[3,5,7],
        temporal_channels=16,
        gcn_channels=32,
        num_classes=2,
        dropout_rate=0.5,
        gat_heads=4,
        attn_hidden=32
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 3) Шкала шумов
    noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
    accs, aucs = test_with_noise(model, dataset, edge_index, device, noise_levels)

    # 4) Построение кривой устойчивости
    plt.figure(figsize=(8,4))
    plt.plot(noise_levels, accs, '-o', label='Accuracy (%)')
    plt.plot(noise_levels, aucs, '-s', label='AUC ×100 (%)')
    plt.xlabel('Noise σ')
    plt.ylabel('Value (%)')
    plt.title('Stability Curve: Accuracy & AUC vs Noise Level')
    plt.xticks(noise_levels)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
