import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from torch_geometric.nn import global_mean_pool
from torch.utils.data import DataLoader

# Импорт вашего определения модели и датасета
from model.RobustHybridEEGModelV3 import (
    HybridEEGGraphModel,
    EEGDataset,
    compute_adjacency,
    create_edge_index,
    CONFIG,
    SFREQ,
    SUBCLIP_LEN
)

if __name__ == "__main__":
    # Параметры
    data_root   = "preproc_clips_test"        # папка с .npz файлами тестовой выборки
    model_path  = "best_model_91%.pt"        # путь к сохранённой модели
    channel_names = [
        'Fp1','Fp2','F7','F3','Fz','F4','F8',
        'T3','C3','Cz','C4','T4','T5','P3',
        'Pz','P4','T6','O1','O2'
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Готовим Dataset
    test_ds    = EEGDataset(data_root, channel_names, augment=False)
    print(f"Всего клипов в тесте: {len(test_ds)}")

    # 2) Строим графовую структуру
    adj        = compute_adjacency(channel_names)
    edge_index = create_edge_index(adj, 0.2).to(device)

    # 3) Загружаем модель (с тем же bp_dim, с которым она обучалась)
    model = HybridEEGGraphModel(
        n_ch=len(channel_names),
        feat_dim=32,
        spat_dim=32,
        bp_dim=8,                # ← bp_dim=4, как при обучении
        gcn_h=128,
        heads=4,
        dropout=CONFIG['dropout']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Модель загружена, режим eval().")

    # 4) Проверка устойчивости к шуму
    noise_levels = np.arange(0.1, 1.0, 0.1)
    accuracies   = []
    aucs         = []

    for sigma in noise_levels:
        preds = []
        trues = []
        probs = []
        latencies = []

        for idx in range(len(test_ds)):
            clip, bp, label = test_ds[idx]
            # добавляем гауссов шум
            noise = torch.randn_like(clip) * sigma
            clip_noisy = clip + noise

            clip_noisy = clip_noisy.unsqueeze(0).to(device)  # [1, C, T]
            bp_tensor  = bp.unsqueeze(0).to(device)
            label_int  = int(label)

            start = time.time()
            with torch.no_grad():
                B, C, T = clip_noisy.shape
                batch_idx = (
                    torch.arange(B, device=device)
                         .unsqueeze(1)
                         .repeat(1, C)
                         .view(-1)
                )
                logits = model(clip_noisy, bp_tensor, edge_index, batch_idx)
            latencies.append(time.time() - start)

            prob = F.softmax(logits, dim=1)[0,1].item()
            pred = logits.argmax(dim=1).item()

            preds.append(pred)
            trues.append(label_int)
            probs.append(prob)

        # метрики для данного σ
        total = len(trues)
        acc   = sum(p==t for p,t in zip(preds, trues)) / total
        fpr, tpr, _ = roc_curve(trues, probs)
        roc_auc     = auc(fpr, tpr)

        accuracies.append(acc)
        aucs.append(roc_auc)

        print(f"σ={sigma:.1f} → Accuracy: {acc*100:.2f}%, AUC: {roc_auc:.3f}, "
              f"avg latency: {np.mean(latencies):.3f}s")

    # 5) Визуализация зависимости качества от шума
    plt.figure(figsize=(6,4))
    plt.plot(noise_levels, np.array(accuracies)*100, marker='o')
    plt.title("Accuracy vs Gaussian Noise σ")
    plt.xlabel("σ")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(noise_levels, aucs, marker='o')
    plt.title("AUC vs Gaussian Noise σ")
    plt.xlabel("σ")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
