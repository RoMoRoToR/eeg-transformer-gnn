import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import seaborn as sns
from torch_geometric.nn import GATConv, global_mean_pool

###########################################
# Класс для загрузки данных EEG
###########################################

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
        # Преобразуем к виду [1, n_channels, n_samples] для Conv2D
        clip = clip[np.newaxis, ...]
        clip = torch.tensor(clip, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return clip, label

###########################################
# Реализация DepthwiseConv2d и SeparableConv2d
###########################################

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=0, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.depthwise(x)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

###########################################
# Модель EEGNet2D с адаптивным пуллингом для бинарной классификации
###########################################

class EEGNet2D(nn.Module):
    """
    Модель использует 2D-свёртки для обработки EEG данных.
    Входной формат: [batch_size, 1, n_channels, n_samples]
    Выход: вероятность (через Sigmoid) для бинарной классификации.
    AdaptiveAvgPool2d фиксирует выходную размерность.
    """
    def __init__(self, n_channels=19, n_samples=500, dropout_rate=0.5):
        super(EEGNet2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,25),
                               padding=(0,12), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.depthwise = DepthwiseConv2d(in_channels=16, kernel_size=(n_channels,1), bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1,4))
        self.dropout1 = nn.Dropout(dropout_rate)
        self.sepconv = SeparableConv2d(in_channels=16, out_channels=16,
                                       kernel_size=(1,16), padding=(0,8), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AdaptiveAvgPool2d((1,15))
        self.dropout2 = nn.Dropout(dropout_rate)
        # Выходной размер: 16 * 1 * 15 = 240
        self.fc = nn.Linear(240, 1)
        self.sigmoid = nn.Sigmoid()

        # Вычисление выходной размерности для проверки
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.conv1(dummy)
            out = self.bn1(out)
            out = self.depthwise(out)
            out = self.bn2(out)
            out = self.elu1(out)
            out = self.pool1(out)
            out = self.dropout1(out)
            out = self.sepconv(out)
            out = self.bn3(out)
            out = self.elu2(out)
            out = self.pool2(out)
            out = self.dropout2(out)
            flat_dim = out.view(1, -1).shape[1]
            print("Computed flat dimension:", flat_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.depthwise(out)
        out = self.bn2(out)
        out = self.elu1(out)
        out = self.pool1(out)
        out = self.dropout1(out)
        out = self.sepconv(out)
        out = self.bn3(out)
        out = self.elu2(out)
        out = self.pool2(out)
        out = self.dropout2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return self.sigmoid(out)

###########################################
# Онлайн-тестирование модели на потоковых данных (без фильтрации)
# с дополнительными графиками для наглядности
###########################################

if __name__ == "__main__":
    data_root = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"
    model_path = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/model/eegnet2d_model_final.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка обученной модели EEGNet2D
    model = EEGNet2D(n_channels=19, n_samples=500, dropout_rate=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Загрузка датасета
    dataset = EEGDataset(data_root)

    print("Начало онлайн-тестирования модели (исходные данные, без фильтрации)...")
    preds_list = []
    labels_list = []
    latencies = []
    probs_list = []  # Для хранения вероятностей, выдаваемых моделью

    for i in range(len(dataset)):
        clip, label = dataset[i]
        clip = clip.unsqueeze(0).to(device)  # [1, 1, n_channels, n_samples]
        label = label.to(device)

        start_time = time.time()
        with torch.no_grad():
            output = model(clip)
        latency = time.time() - start_time
        latencies.append(latency)

        probability = output.item()
        probs_list.append(probability)
        pred = 1 if probability >= 0.5 else 0
        preds_list.append(pred)
        labels_list.append(label.item())

        print(f"Клип {i+1}/{len(dataset)}: предсказано {pred}, истинное {label.item()}, время обработки: {latency:.4f} сек.")

    # Рассчет общей точности и времени обработки
    correct = sum(1 for p, l in zip(preds_list, labels_list) if p == l)
    total = len(labels_list)
    accuracy = 100.0 * correct / total
    avg_latency = sum(latencies) / total
    print(f"\nОнлайн-тестирование завершено.")
    print(f"Общая точность: {accuracy:.2f}%")
    print(f"Среднее время обработки одного клипа: {avg_latency:.4f} сек.")

    # --- Дополнительные графики для наглядности ---

    # 1. Confusion Matrix и Classification Report
    cm = confusion_matrix(labels_list, preds_list)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(labels_list, preds_list, target_names=["Class 0", "Class 1"]))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    # 2. График задержек обработки (latency per clip)
    plt.figure()
    plt.plot(range(1, total + 1), latencies, marker='o')
    plt.title("Latency per clip (online processing, raw data)")
    plt.xlabel("Clip index")
    plt.ylabel("Latency (sec)")
    plt.grid(True)
    plt.show()

    # 3. Гистограмма распределения задержек
    plt.figure()
    plt.hist(latencies, bins=20, color='skyblue', edgecolor='black')
    plt.title("Histogram of Latencies")
    plt.xlabel("Latency (sec)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # 4. ROC-кривая
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(labels_list, probs_list)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # 5. Гистограмма распределения вероятностей модели
    plt.figure()
    plt.hist(probs_list, bins=20, color='lightgreen', edgecolor='black')
    plt.title("Distribution of Predicted Probabilities")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
