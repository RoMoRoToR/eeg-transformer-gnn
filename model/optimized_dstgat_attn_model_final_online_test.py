import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from torch_geometric.nn import global_mean_pool
from optimized_dstgat_attn_model_final import Optimized_DSTGAT_Attn, EEGDataset, compute_adjacency, create_edge_index
# Предполагается, что модель Optimized_DSTGAT_Attn и функции создания графа импортированы или определены выше.
# Например, импортируем:
# from your_model_definition_file import Optimized_DSTGAT_Attn, EEGDataset, compute_adjacency, create_edge_index

if __name__ == "__main__":
    # Задание путей и параметров
    data_root = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"  # путь к данным
    model_path = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/model/optimized_dstgat_attn_model_final.pth"  # путь к сохранённой модели
    n_channels = 19
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка датасета
    dataset = EEGDataset(data_root)
    print(f"Найдено клипов: {len(dataset)}")

    # Вычисляем матрицу смежности и формируем edge_index для графовых слоёв
    adj = compute_adjacency(n_channels)
    edge_index = create_edge_index(adj, threshold=0.3).to(device)

    # Загружаем модель и её веса
    model = Optimized_DSTGAT_Attn(
        n_channels=n_channels,
        eeg_feat_dim=16,
        gcn_channels=32,
        num_classes=2,
        dropout_rate=0.5,
        gat_heads=4
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Перевод модели в режим оценки (inference)
    print("Модель загружена и переведена в режим eval().")

    # Списки для накопления результатов
    preds_list = []
    labels_list = []
    latencies = []
    probs_list = []  # для хранения вероятностей класса 1 (ROC-кривая)

    print("Начало онлайн-тестирования модели...")
    # Проход по всем клипам (обработка по одному)
    for idx in range(len(dataset)):
        clip, label = dataset[idx]
        # Добавляем батчевую размерность: [1, n_channels, n_samples]
        clip = clip.unsqueeze(0).to(device)
        label = label.item()

        start_time = time.time()
        with torch.no_grad():
            output = model(clip, edge_index)
        elapsed_time = time.time() - start_time
        latencies.append(elapsed_time)

        # Получаем предсказание и вероятность класса 1
        _, pred = torch.max(output, dim=1)
        preds_list.append(pred.item())
        labels_list.append(label)
        prob = F.softmax(output, dim=1)[0, 1].item()  # вероятность для класса 1
        probs_list.append(prob)

        print(f"Клип {idx + 1}/{len(dataset)}: предсказано {pred.item()}, истинное {label}, время обработки: {elapsed_time:.4f} сек.")

    # Итоговые метрики
    correct = sum(int(p == l) for p, l in zip(preds_list, labels_list))
    total = len(labels_list)
    accuracy = 100.0 * correct / total
    avg_latency = np.mean(latencies)
    print("\nОнлайн-тестирование завершено.")
    print(f"Общая точность: {accuracy:.2f}%")
    print(f"Среднее время обработки одного клипа: {avg_latency:.4f} сек.")

    # Вычисление матрицы ошибок и вывод отчёта по классификации
    cm = confusion_matrix(labels_list, preds_list)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(labels_list, preds_list, target_names=["Class 0", "Class 1"]))

    # Визуализация результатов

    # 1. Матрица ошибок
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Class 0", "Class 1"])
    plt.yticks(tick_marks, ["Class 0", "Class 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    # 2. График задержек обработки (latency per clip)
    plt.figure()
    plt.plot(range(1, total + 1), latencies, marker='o')
    plt.xlabel("Номер клипа")
    plt.ylabel("Время обработки (сек)")
    plt.title("Latency per Clip (Онлайн-тестирование)")
    plt.grid(True)
    plt.show()

    # 3. Гистограмма распределения задержек
    plt.figure()
    plt.hist(latencies, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Время обработки (сек)")
    plt.ylabel("Частота")
    plt.title("Гистограмма задержек обработки")
    plt.grid(True)
    plt.show()

    # 4. ROC-кривая
    fpr, tpr, _ = roc_curve(labels_list, probs_list)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # 5. Гистограмма распределения вероятностей предсказаний
    plt.figure()
    plt.hist(probs_list, bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel("Вероятность (Класс 1)")
    plt.ylabel("Частота")
    plt.title("Распределение вероятностей предсказаний")
    plt.grid(True)
    plt.show()
