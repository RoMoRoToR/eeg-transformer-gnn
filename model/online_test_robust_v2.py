import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from torch.utils.data import Dataset

# Импортируем вашу улучшенную модель и утилиты
from RobustHybridEEGModelV2 import (
    RobustHybridEEGModelV2,
    EEGDataset,
    compute_adjacency,
    create_edge_index
)

if __name__ == "__main__":
    ############################
    # Настройки
    ############################
    data_root  = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips_no_filter"
    model_path = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/model/best_model_v2.pth"
    n_channels = 19
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################
    # Загрузка модели
    ############################
    model = RobustHybridEEGModelV2(
        n_channels=n_channels,
        n_samples=500,
        feat_dim=16,
        gcn_hidden=64,
        heads=8,
        num_classes=2,
        noise_sigma=0.0,   # при тесте без добавления шума
        dropout=0.0        # отключаем дропаут при инференсе
    ).to(device)
    # 1) Загрузим чекпойнт
    loaded_ckpt = torch.load(model_path, map_location=device)

    # 2) Вытянем state_dict текущей модели
    model_dict = model.state_dict()

    # 3) Отфильтруем только совпадающие по имени и по форме параметры
    filtered_ckpt = {}
    for k, v in loaded_ckpt.items():
        if k in model_dict and v.size() == model_dict[k].size():
            filtered_ckpt[k] = v
        else:
            # вы можете раскомментировать следующую строку,
            # чтобы увидеть, какие ключи пропускаются
            # print(f"SKIP {k}: loaded shape={v.size()} vs model shape={model_dict.get(k, 'MISSING')}")
            pass

    # 4) Объединим и загрузим
    model_dict.update(filtered_ckpt)
    model.load_state_dict(model_dict)

    print(f"✅ Loaded {len(filtered_ckpt)}/{len(model_dict)} parameters from checkpoint")
    model.eval()

    ############################
    # Графовые данные
    ############################
    adj        = compute_adjacency(n_channels)
    edge_index = create_edge_index(adj, threshold=0.3).to(device)

    ############################
    # Датасет
    ############################
    dataset = EEGDataset(data_root, augment=False)

    ############################
    # Онлайн-тест
    ############################
    preds_list, labels_list, probs_list, latencies = [], [], [], []

    print("Начало онлайн-тестирования RobustHybridEEGModelV2...")
    for i, (clip, label) in enumerate(dataset):
        # clip: [C, T] → [1, C, T]
        clip = clip.unsqueeze(0).to(device)
        label = torch.tensor(label, device=device)

        # batch_index для графового блока: [0,0,...,0] длины C
        B = clip.size(0)
        batch_idx = torch.arange(B, device=device)\
                        .unsqueeze(1)\
                        .repeat(1, n_channels)\
                        .view(-1)

        start = time.time()
        with torch.no_grad():
            logits = model(clip, edge_index, batch_idx)
        latency = time.time() - start

        prob = F.softmax(logits, dim=1)[0,1].item()
        pred = logits.argmax(dim=1)[0].item()

        latencies.append(latency)
        probs_list.append(prob)
        preds_list.append(pred)
        labels_list.append(label.item())

        print(f"[{i+1}/{len(dataset)}] Предсказание: {pred}, Истинно: {label.item()}, Время: {latency:.4f} с")

    ############################
    # Итоговые метрики
    ############################
    accuracy    = 100.0 * np.mean([p==l for p,l in zip(preds_list, labels_list)])
    avg_latency = np.mean(latencies)

    print(f"\nОнлайн-тест завершён.")
    print(f"Точность: {accuracy:.2f}%")
    print(f"Средняя задержка: {avg_latency:.4f} сек")

    ############################
    # 1. Confusion Matrix & Classification Report
    ############################
    cm = confusion_matrix(labels_list, preds_list)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n",
          classification_report(labels_list, preds_list, target_names=["Class 0","Class 1"]))

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0","1"], yticklabels=["0","1"])
    plt.title("Confusion Matrix")
    plt.xlabel("Предсказано")
    plt.ylabel("Истинно")
    plt.show()

    ############################
    # 2. Latency per Clip
    ############################
    plt.figure()
    plt.plot(range(1,len(latencies)+1), latencies, marker='o')
    plt.title("Задержка обработки по клипам")
    plt.xlabel("Индекс клипа")
    plt.ylabel("Latency (сек)")
    plt.grid(True)
    plt.show()

    ############################
    # 3. Гистограмма Latencies
    ############################
    plt.figure()
    plt.hist(latencies, bins=20, edgecolor='black')
    plt.title("Гистограмма задержек")
    plt.xlabel("Latency (сек)")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.show()

    ############################
    # 4. ROC-кривая
    ############################
    fpr, tpr, _   = roc_curve(labels_list, probs_list)
    roc_auc       = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],"k--")
    plt.title("ROC-кривая")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    ############################
    # 5. Распределение вероятностей
    ############################
    plt.figure()
    plt.hist(probs_list, bins=20, edgecolor='black')
    plt.title("Распределение предсказанных вероятностей (класс 1)")
    plt.xlabel("Probability")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.show()

    ############################
    # 6. Задержка vs Вероятность
    ############################
    plt.figure()
    plt.scatter(probs_list, latencies, alpha=0.6)
    plt.title("Latency vs Predicted Probability")
    plt.xlabel("Predicted Probability (Class 1)")
    plt.ylabel("Latency (сек)")
    plt.grid(True)
    plt.show()
