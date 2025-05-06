import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

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
# Модель EEGNet2D с адаптивным пуллингом
###########################################

class EEGNet2D(nn.Module):
    """
    Модель использует 2D-свёртки для обработки EEG данных.
    Входной формат: [batch_size, 1, n_channels, n_samples]
    Выход: вероятность (sigmoid) для бинарной классификации.
    Благодаря использованию AdaptiveAvgPool2d выходная размерность фиксирована.
    """
    def __init__(self, n_channels=19, n_samples=500, dropout_rate=0.5):
        super(EEGNet2D, self).__init__()
        # Первый Conv2D
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,25),
                               padding=(0,12), bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # DepthwiseConv2D: свёртка по пространственной оси (используем n_channels)
        self.depthwise = DepthwiseConv2d(in_channels=16, kernel_size=(n_channels,1), bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.elu1 = nn.ELU()

        # AveragePooling (фиксированное уменьшение по временной оси)
        self.pool1 = nn.AvgPool2d(kernel_size=(1,4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # SeparableConv2D
        self.sepconv = SeparableConv2d(in_channels=16, out_channels=16,
                                       kernel_size=(1,16), padding=(0,8), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.elu2 = nn.ELU()

        # Adaptive pooling для получения фиксированного размера по временной оси
        self.pool2 = nn.AdaptiveAvgPool2d((1,15))
        self.dropout2 = nn.Dropout(dropout_rate)

        # После AdaptiveAvgPool2d выход имеет размер: [B, 16, 1, 15]
        # => flat_dim = 16*1*15 = 240
        self.fc = nn.Linear(16*1*15, 1)
        self.sigmoid = nn.Sigmoid()

        # Можно вывести для проверки:
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
        # x: [batch_size, 1, n_channels, n_samples]
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
# Функция обучения и оценки модели
###########################################

def train_model(model, train_loader, test_loader, device, num_epochs=50, lr=1e-3):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * clips.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        model.eval()
        test_running_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for clips, labels in test_loader:
                clips, labels = clips.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)
                outputs = model(clips)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item() * clips.size(0)
                preds = (outputs >= 0.5).float()
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)
        test_loss = test_running_loss / total_test
        test_acc = correct_test / total_test
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step(test_loss)
        print(f"Эпоха {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    return train_losses, train_accs, test_losses, test_accs

def plot_curves(train_losses, train_accs, test_losses, test_accs):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('Кривые потерь')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, test_accs, label='Test Acc')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.title('Кривые точности')
    plt.legend()
    plt.tight_layout()
    plt.show()

###########################################
# Основной блок: обучение модели
###########################################

if __name__ == "__main__":
    data_root = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EEGDataset(data_root)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_data, test_data = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    model = EEGNet2D(n_channels=19, n_samples=500, dropout_rate=0.5)
    model.to(device)

    train_losses, train_accs, test_losses, test_accs = train_model(
        model, train_loader, test_loader, device, num_epochs=50, lr=1e-3
    )
    plot_curves(train_losses, train_accs, test_losses, test_accs)

    torch.save(model.state_dict(), "eegnet2d_model_final.pth")
    print("Обучение завершено, модель сохранена.")
