import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import GATConv, global_mean_pool
import matplotlib.pyplot as plt


###########################################
# 1D-модули на основе EEGNet для временной обработки
###########################################

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=0, bias=False):
        super(DepthwiseConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.depthwise(x)


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super(SeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGFeatureExtractor1D(nn.Module):
    """
    Экстрактор временных признаков для каждого электрода, построенный по принципу модулей EEGNet.
    Вход: [B*n_channels, 1, n_samples]
    Выход: [B*n_channels, feat_dim] – вектор признаков для дальнейшей графовой обработки
    """

    def __init__(self, temporal_filters=16, output_dim=16, kernel_size=25, pooling_kernel=4, dropout_rate=0.5):
        super(EEGFeatureExtractor1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=temporal_filters,
                               kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(temporal_filters)
        self.elu = nn.ELU()

        self.depthwise = DepthwiseConv1d(temporal_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(temporal_filters)

        self.separable = SeparableConv1d(temporal_filters, output_dim, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(output_dim)

        self.pool = nn.AvgPool1d(kernel_size=pooling_kernel)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [B*n_channels, 1, n_samples]
        x = self.conv1(x)  # -> [B*n_channels, temporal_filters, n_samples]
        x = self.bn1(x)
        x = self.elu(x)

        x = self.depthwise(x)  # -> [B*n_channels, temporal_filters, n_samples]
        x = self.bn2(x)
        x = self.elu(x)

        x = self.separable(x)  # -> [B*n_channels, output_dim, n_samples]
        x = self.bn3(x)
        x = self.elu(x)

        x = self.pool(x)  # Снижение размерности по времени
        x = self.dropout(x)
        x = F.adaptive_avg_pool1d(x, 1)  # Получаем [B*n_channels, output_dim, 1]
        x = x.squeeze(-1)  # [B*n_channels, output_dim]
        return x


###########################################
# Функции для работы с графовыми структурами
###########################################

def create_edge_index(adj_matrix, threshold=0.1):
    src, dst = np.where((adj_matrix > threshold) & (np.eye(adj_matrix.shape[0]) == 0))
    edge_index = np.vstack((src, dst))
    return torch.tensor(edge_index, dtype=torch.long)


def compute_adjacency(n_channels):
    # Используются случайные координаты; при наличии реальных координат заменить генерацию
    coords = np.random.rand(n_channels, 2)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    eps = 1e-5
    adj = 1.0 / (dist + eps)
    np.fill_diagonal(adj, 0)
    adj = adj / np.max(adj)
    return adj


###########################################
# Оптимизированная модель: объединение EEGNet-модуля для временной обработки и GAT для пространственной агрегации
###########################################

class Optimized_DSTGAT_Attn(nn.Module):
    """
    Модель объединяет эффективный EEGNet-блок для извлечения временных признаков с графовыми
    слоями (GATConv) для моделирования связей между каналами.

    Вход: [batch_size, n_channels, n_samples]
    Выход: логиты для классификации (num_classes)
    """

    def __init__(self, n_channels, eeg_feat_dim=16, gcn_channels=32, num_classes=2,
                 dropout_rate=0.5, gat_heads=4):
        super(Optimized_DSTGAT_Attn, self).__init__()
        self.n_channels = n_channels
        self.feature_extractor = EEGFeatureExtractor1D(temporal_filters=16, output_dim=eeg_feat_dim,
                                                       kernel_size=25, pooling_kernel=4, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

        # Первый графовый слой с residual-подключением
        self.gat1 = GATConv(eeg_feat_dim, gcn_channels, heads=gat_heads, concat=False)
        self.gat1_bn = nn.BatchNorm1d(gcn_channels)
        self.skip1 = nn.Linear(eeg_feat_dim, gcn_channels)

        # Второй графовый слой с residual-подключением
        self.gat2 = GATConv(gcn_channels, gcn_channels, heads=gat_heads, concat=False)
        self.gat2_bn = nn.BatchNorm1d(gcn_channels)

        self.fc = nn.Linear(gcn_channels, num_classes)

    def forward(self, x, edge_index):
        # x: [batch_size, n_channels, n_samples]
        batch_size, n_channels, n_samples = x.shape

        # Применяем экстракцию временных признаков для каждого канала
        x = x.view(batch_size * n_channels, 1, n_samples)  # [B*n_channels, 1, n_samples]
        x = self.feature_extractor(x)  # [B*n_channels, eeg_feat_dim]

        # Переформатируем узловые признаки для дальнейшей графовой обработки
        x = x.view(batch_size, n_channels, -1)  # [B, n_channels, eeg_feat_dim]
        x = x.view(batch_size * n_channels, -1)

        # Корректировка edge_index для пакета (сдвиг индексов для каждого примера)
        new_edge_index = []
        for i in range(batch_size):
            new_edge_index.append(edge_index + i * n_channels)
        new_edge_index = torch.cat(new_edge_index, dim=1).to(x.device)
        batch_vector = torch.arange(batch_size, device=x.device).unsqueeze(1).repeat(1, n_channels).view(-1)

        # Первый графовый слой с residual-соединением
        x_in = x
        x = self.gat1(x, new_edge_index)
        x = self.gat1_bn(x)
        x_skip = self.skip1(x_in)
        x = F.gelu(x + x_skip)
        x = self.dropout(x)

        # Второй графовый слой с residual-соединением
        x_in2 = x
        x = self.gat2(x, new_edge_index)
        x = self.gat2_bn(x)
        x = F.gelu(x + x_in2)
        x = self.dropout(x)

        # Глобальная агрегация (среднее по узлам) и классификация
        x = global_mean_pool(x, batch_vector)
        logits = self.fc(x)
        return logits


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
        clip = torch.tensor(clip, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return clip, label


###########################################
# Функция обучения и отображения результатов
###########################################

def train_model(model, train_loader, test_loader, edge_index, device, num_epochs=50, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for clips, labels in train_loader:
            clips = clips.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(clips, edge_index.to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * clips.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for clips, labels in test_loader:
                clips = clips.to(device)
                labels = labels.to(device)
                outputs = model(clips, edge_index.to(device))
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * clips.size(0)
                _, preds = torch.max(outputs, 1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)
        test_loss = running_test_loss / total_test
        test_acc = correct_test / total_test
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step(test_loss)
        print(f"Эпоха {epoch + 1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    return train_losses, train_accs, test_losses, test_accs


def plot_curves(train_losses, train_accs, test_losses, test_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('Кривые потерь')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, test_accs, label='Test Acc')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.title('Кривые точности')
    plt.legend()
    plt.tight_layout()
    plt.show()


###########################################
# Основной блок: подготовка данных, вычисление матрицы смежности и обучение модели
###########################################

if __name__ == "__main__":
    data_root = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"  # замените на ваш путь к данным
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EEGDataset(data_root)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_data, test_data = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # Вычисляем матрицу смежности для графовых слоев
    n_channels = 19
    adj = compute_adjacency(n_channels)
    edge_index = create_edge_index(adj, threshold=0.3).to(device)

    # Инициализируем оптимизированную модель
    model = Optimized_DSTGAT_Attn(n_channels=n_channels, eeg_feat_dim=16,
                                  gcn_channels=32, num_classes=2, dropout_rate=0.5, gat_heads=4)
    model.to(device)

    train_losses, train_accs, test_losses, test_accs = train_model(
        model, train_loader, test_loader, edge_index, device, num_epochs=50, lr=1e-3
    )
    plot_curves(train_losses, train_accs, test_losses, test_accs)

    torch.save(model.state_dict(), "optimized_dstgat_attn_model_final.pth")
    print("Обучение завершено, модель сохранена.")
