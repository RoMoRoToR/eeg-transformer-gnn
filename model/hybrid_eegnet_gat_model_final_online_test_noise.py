import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score
from torch_geometric.nn import global_mean_pool

# Импорт вашей модели и датасета
from hybrid_eegnet_gat_model_final import HybridEEG_GNN, EEGDataset, compute_adjacency, create_edge_index

def evaluate_with_noise(model, dataset, edge_index, device, noise_levels):
    """
    Для каждого σ в noise_levels добавляет к каждому клипу шум N(0, σ^2),
    прогоняет модель, собирает Accuracy и AUC.
    """
    accs = []
    aucs = []
    for sigma in noise_levels:
        preds, labels, probs = [], [], []
        for clip, label in dataset:
            clip = clip.unsqueeze(0).to(device)  # [1,1,C,T]
            # добавляем шум
            noise = torch.randn_like(clip) * sigma
            clip_noisy = clip + noise

            with torch.no_grad():
                # строим batch_index
                B, _, C, T = clip_noisy.shape
                batch_index = torch.arange(B, device=device)\
                                  .unsqueeze(1).repeat(1, C).view(-1)
                out = model(clip_noisy, edge_index, batch_index)
                prob = F.softmax(out, dim=1)[0,1].item()
                pred = out.argmax(dim=1).item()

            preds.append(pred)
            labels.append(label.item())
            probs.append(prob)

        # метрики
        acc = accuracy_score(labels, preds) * 100
        fpr, tpr, _ = roc_curve(labels, probs)
        model_auc = auc(fpr, tpr) * 100
        accs.append(acc)
        aucs.append(model_auc)
        print(f"σ={sigma:.3f} → Acc={acc:.1f}%, AUC={model_auc:.1f}%")
    return accs, aucs

if __name__ == "__main__":
    # Параметры
    data_root = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"
    model_path = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/model/hybrid_eegnet_gat_model_final.pth"
    n_channels = 19
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Датасет и модель
    dataset = EEGDataset(data_root)
    adj = compute_adjacency(n_channels)
    edge_index = create_edge_index(adj, threshold=0.3).to(device)

    model = HybridEEG_GNN(n_channels=n_channels, n_samples=500,
                          dropout_rate=0.5, gcn_channels=32,
                          gat_heads=4, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Уровни шума для теста
    noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]

    # Запуск оценки
    accs, aucs = evaluate_with_noise(model, dataset, edge_index, device, noise_levels)

    # Визуализация «кривой устойчивости»
    plt.figure(figsize=(8,5))
    sns.set_style("whitegrid")

    ax = plt.gca()
    ax2 = ax.twinx()

    ax.plot(noise_levels, accs, '-o', label='Accuracy (%)', color='tab:blue')
    ax2.plot(noise_levels, aucs, '-s', label='AUC ×100 (%)', color='tab:orange')

    ax.set_xlabel('Noise σ', fontsize=12)
    ax.set_ylabel('Accuracy (%)', color='tab:blue', fontsize=12)
    ax2.set_ylabel('AUC ×100 (%)', color='tab:orange', fontsize=12)
    plt.title('Stability Curve: Accuracy & AUC vs Noise Level', fontsize=14)

    ax.set_xticks(noise_levels)
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    # легенда
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines+lines2, labels+labels2, loc='lower left')

    plt.tight_layout()
    plt.show()
