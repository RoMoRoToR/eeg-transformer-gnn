import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter


# ------------------ Кастомный Dataset для NPZ-файлов ------------------
class EEGClipDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: корневая папка с предобработанными клипами.
        Ожидаемая структура:
            preproc_clips/
                sub-001/
                    sub-001_clip_0.npz
                    sub-001_clip_1.npz
                    ...
                sub-002/
                    sub-002_clip_0.npz
                    ...
        """
        self.file_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".npz"):
                    self.file_paths.append(os.path.join(subdir, file))
        self.file_paths.sort()  # для консистентности

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npz_path = self.file_paths[idx]
        data_npz = np.load(npz_path)
        # Ожидаем, что в NPZ-файле содержатся:
        #   clip: массив размера (num_channels, n_samples)
        #   sfreq: частота дискретизации (число)
        #   label: метка (0 или 1)
        clip = data_npz['clip']  # (C, T)
        label = int(data_npz['label'])
        # Преобразуем в тензор типа float
        clip_tensor = torch.tensor(clip, dtype=torch.float)
        return clip_tensor, label


# ------------------ CNN-модель для EEG ------------------
class EEG_CNN_1D(nn.Module):
    def __init__(self, num_channels, num_classes):
        """
        Пример CNN для EEG, использующий 1D свёртки по временной оси.
        Вход: тензор размера (batch, num_channels, n_samples).
        """
        super(EEG_CNN_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        # Используем адаптивный пуллинг, чтобы получить фиксированное представление независимо от длины сигнала
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = Linear(64, num_classes)

    def forward(self, x):
        # x имеет размер (batch, num_channels, n_samples)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.avgpool(x)  # (batch, 64, 1)
        x = x.squeeze(-1)  # (batch, 64)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, num_classes)
        return F.log_softmax(x, dim=-1)


# ------------------ Обучение и оценка ------------------
def train_one_epoch(model, loader, optimizer, device, weights_tensor=None):
    model.train()
    total_loss = 0.0
    for data, labels in loader:
        data = data.to(device)  # data shape: (batch, channels, time)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, labels,
                               weight=weights_tensor) if weights_tensor is not None else F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for data, labels in loader:
        data = data.to(device)
        labels = labels.to(device)
        output = model(data)
        preds = output.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)


# ------------------ Основной скрипт ------------------
if __name__ == "__main__":
    # Укажите путь к вашей директории с предобработанными клипами
    root_dir = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"

    # Создаем Dataset
    dataset = EEGClipDataset(root_dir)
    print(f"Найдено клипов: {len(dataset)}")

    # Получаем метки для стратифицированного разбиения
    labels_all = [dataset[i][1] for i in range(len(dataset))]
    num_classes = len(set(labels_all))
    print("Уникальные метки:", set(labels_all))

    # Разбиваем на train/test (например, 80/20)
    all_data = [dataset[i] for i in range(len(dataset))]
    train_data, test_data = train_test_split(all_data, test_size=0.2, shuffle=True, stratify=labels_all,
                                             random_state=42)
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    # Создаем DataLoader-ы
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Подсчет классов и вычисление весов
    count_train = Counter([data[1] for data in train_data])
    print("Train class distribution:", count_train)
    weights = np.array([1.0 / count_train.get(c, 1) for c in range(num_classes)], dtype=np.float32)
    weights /= weights.sum()
    weights_tensor = torch.tensor(weights, dtype=torch.float, device=device)
    print("Class weights:", weights)

    # Определяем параметры: количество каналов (например, 19) и число классов
    sample, _ = dataset[0]
    num_channels = sample.shape[0]
    print(f"num_channels = {num_channels}, num_classes = {num_classes}")

    # Создаем модель
    model = EEG_CNN_1D(num_channels=num_channels, num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, verbose=True)

    epochs = 100
    best_val_acc = 0.0
    times_epoch = []

    # Если у вас нет отдельной валидационной выборки, можно использовать test_data для оценки.
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, weights_tensor=weights_tensor)
        test_acc = eval_model(model, test_loader, device)
        scheduler.step(test_acc)
        end_time = time.time()
        times_epoch.append(end_time - start_time)
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            best_model_state = model.state_dict().copy()
        print(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | TestAcc: {test_acc:.4f} | Best: {best_val_acc:.4f}")

    avg_time = np.mean(times_epoch)
    print(f"\nСреднее время эпохи: {avg_time:.4f} сек ({1.0 / avg_time:.2f} эпох/с)")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_acc = eval_model(model, test_loader, device)
    print(f"Final Test Accuracy: {final_acc:.4f}")
