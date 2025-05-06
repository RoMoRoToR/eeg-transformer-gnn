import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


###########################################
# Функции для работы с графами
###########################################
def compute_adjacency(n_channels):
    """
    Вычисляет матрицу смежности на основе случайных координат электродов.
    При наличии реальных координат (схема 10-20) следует заменить генерацию.
    """
    coords = np.random.rand(n_channels, 2)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    eps = 1e-5
    adj = 1.0 / (dist + eps)
    np.fill_diagonal(adj, 0)
    adj = adj / np.max(adj)
    return adj


def create_edge_index(adj_matrix, threshold=0.1):
    """
    Создает edge_index на основе матрицы смежности.
    Отбираются только те ребра, где значение больше порога и не учитываются диагональные элементы.
    """
    src, dst = np.where((adj_matrix > threshold) & (np.eye(adj_matrix.shape[0]) == 0))
    edge_index = np.vstack((src, dst))
    return torch.tensor(edge_index, dtype=torch.long)


###########################################
# Блок извлечения признаков EEGNet2D (модифицированная версия)
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


class EEGNet2D_Hybrid(nn.Module):
    """
    Модель извлечения признаков EEGNet2D_Hybrid: извлекаются признаки по каждому каналу.
    Входной формат: [B, 1, n_channels, n_samples].
    Выход: [B, n_channels, feat_dim] – каждый канал представлен вектором признаков.
    """

    def __init__(self, n_channels=19, n_samples=500, dropout_rate=0.5):
        super(EEGNet2D_Hybrid, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 25),
                               padding=(0, 12), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.elu = nn.ELU()
        # Глубинная свертка: обрабатывает каждый канал отдельно
        self.depthwise = DepthwiseConv2d(in_channels=16, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        # Разделённая свёртка для извлечения временных зависимостей
        self.separable = SeparableConv2d(in_channels=16, out_channels=16, kernel_size=(1, 16),
                                         padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        # Адаптивное пуллирование сохраняет размерность по каналам
        self.pool = nn.AdaptiveAvgPool2d((n_channels, 1))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [B, 1, n_channels, n_samples]
        x = self.conv1(x)  # -> [B, 16, n_channels, n_samples]
        x = self.bn1(x)
        x = self.elu(x)
        x = self.depthwise(x)  # -> [B, 16, n_channels, n_samples]
        x = self.bn2(x)
        x = self.elu(x)
        x = self.separable(x)  # -> [B, 16, n_channels, n_samples]
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool(x)  # -> [B, 16, n_channels, 1]
        x = self.dropout(x)
        x = x.squeeze(-1)  # -> [B, 16, n_channels]
        # Переставляем оси для получения [B, n_channels, 16]
        x = x.permute(0, 2, 1)
        return x


###########################################
# Графовый блок (с использованием GATConv) с residual-соединениями
###########################################
class GraphBlock(nn.Module):
    def __init__(self, in_features, gcn_channels, gat_heads, dropout_rate=0.5):
        super(GraphBlock, self).__init__()
        self.gat1 = GATConv(in_features, gcn_channels, heads=gat_heads, concat=False)
        self.bn1 = nn.BatchNorm1d(gcn_channels)
        self.skip1 = nn.Linear(in_features, gcn_channels)
        self.gat2 = GATConv(gcn_channels, gcn_channels, heads=gat_heads, concat=False)
        self.bn2 = nn.BatchNorm1d(gcn_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch_index):
        # x: [B * n_channels, in_features]
        x_in = x
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.gelu(x + self.skip1(x_in))
        x = self.dropout(x)
        x_in2 = x
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.gelu(x + x_in2)
        x = self.dropout(x)
        # Глобальное среднее по узлам для каждого примера
        x = global_mean_pool(x, batch_index)
        return x


###########################################
# Гибридная модель HybridEEG_GNN (объединение EEGNet2D_Hybrid и GraphBlock)
###########################################
class HybridEEG_GNN(nn.Module):
    def __init__(self, n_channels=19, n_samples=500, dropout_rate=0.5,
                 gcn_channels=32, gat_heads=4, num_classes=2):
        super(HybridEEG_GNN, self).__init__()
        self.n_channels = n_channels
        # Извлечение временных признаков с сохранением по каналу
        self.feature_extractor = EEGNet2D_Hybrid(n_channels=n_channels, n_samples=n_samples, dropout_rate=dropout_rate)
        # Графовый блок для пространственной агрегации
        self.graph_block = GraphBlock(in_features=16, gcn_channels=gcn_channels,
                                      gat_heads=gat_heads, dropout_rate=dropout_rate)
        # Финальный классификатор
        self.classifier = nn.Linear(gcn_channels, num_classes)

    def forward(self, x, edge_index, batch_index):
        # x: [B, 1, n_channels, n_samples]
        B = x.size(0)
        features = self.feature_extractor(x)  # -> [B, n_channels, 16]
        # Преобразуем в матрицу узлов: [B*n_channels, 16]
        batch_size, n_channels, feat_dim = features.shape
        features = features.reshape(batch_size * n_channels, feat_dim)
        # Графовая обработка с учетом принадлежности узлов (batch_index)
        graph_out = self.graph_block(features, edge_index, batch_index)  # -> [B, gcn_channels]
        logits = self.classifier(graph_out)
        return logits


###########################################
# Dataset для EEG данных
###########################################
class EEGDataset(Dataset):
    def __init__(self, root_dir):
        """
        Ожидается структура: root_dir/subjectX/*.npz,
        где каждый .npz содержит:
          - 'clip': массив формы [n_channels, n_samples]
          - 'label': метка
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
        clip = data['clip']  # [n_channels, n_samples]
        label = data['label']
        clip = torch.tensor(clip, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        # Для гибридной модели добавляем измерение канала: [1, n_channels, n_samples]
        clip = clip.unsqueeze(0)
        return clip, label


###########################################
# Функции для обучения и онлайн-тестирования с графиками
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
            batch_size = clips.size(0)
            # Для каждого примера повторяем номер примера n_channels раз
            batch_index = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, model.n_channels).reshape(-1)
            outputs = model(clips, edge_index.to(device), batch_index)
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
                batch_size = clips.size(0)
                batch_index = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, model.n_channels).reshape(
                    -1)
                outputs = model(clips, edge_index.to(device), batch_index)
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
        print(
            f"Эпоха {epoch + 1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
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
# Онлайн-тестирование модели HybridEEG_GNN с графиками и дополнительными метриками
###########################################
if __name__ == "__main__":
    data_root = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"  # Замените на актуальный путь к данным
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Вычисление матрицы смежности и формирование edge_index для графового слоя
    n_channels = 19
    adj = compute_adjacency(n_channels)
    edge_index = create_edge_index(adj, threshold=0.3).to(device)

    # Загрузка датасета
    dataset = EEGDataset(data_root)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_data, test_data = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # Инициализация гибридной модели
    model = HybridEEG_GNN(n_channels=n_channels, n_samples=500, dropout_rate=0.5,
                          gcn_channels=32, gat_heads=4, num_classes=2)
    model.to(device)

    # Обучение модели
    train_losses, train_accs, test_losses, test_accs = train_model(
        model, train_loader, test_loader, edge_index, device, num_epochs=50, lr=1e-3
    )
    plot_curves(train_losses, train_accs, test_losses, test_accs)

    # Сохранение модели
    torch.save(model.state_dict(), "hybrid_eegnet_gat_model_final.pth")
    print("Обучение завершено, модель сохранена.")

    # Онлайн-тестирование
    print("Начало онлайн-тестирования модели HybridEEG_GNN...")
    preds_list = []
    labels_list = []
    latencies = []
    probs_list = []  # Для ROC-кривой

    model.eval()
    # Последовательно обрабатываем каждый клип, имитируя потоковую обработку
    for i in range(len(dataset)):
        clip, label = dataset[i]
        clip = clip.unsqueeze(0).to(device)  # [1, 1, n_channels, n_samples]
        label = label.to(device)
        # Для гибридной модели создаём batch_index: для каждого примера повторяем его n_channels раз
        batch_size = clip.size(0)
        batch_index = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, model.n_channels).reshape(-1)

        start_time = time.time()
        with torch.no_grad():
            output = model(clip, edge_index, batch_index)
        latency = time.time() - start_time
        latencies.append(latency)

        # Для CrossEntropyLoss вывод имеет форму [1, num_classes]
        prob = F.softmax(output, dim=1)[0, 1].item()
        probs_list.append(prob)
        _, pred = torch.max(output, 1)
        preds_list.append(pred.item())
        labels_list.append(label.item())

        print(
            f"Клип {i + 1}/{len(dataset)}: Предсказано {pred.item()}, Истинное {label.item()}, Время: {latency:.4f} сек.")

    overall_accuracy = 100.0 * sum(1 for p, l in zip(preds_list, labels_list) if p == l) / len(labels_list)
    avg_latency = np.mean(latencies)
    print(f"\nОнлайн-тестирование завершено. Общая точность: {overall_accuracy:.2f}%")
    print(f"Среднее время обработки клипа: {avg_latency:.4f} сек.")

    # Построение графиков для оценки модели

    # 1. Конфузионная матрица и классификационный отчет
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
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # 2. График задержек обработки по клипам
    plt.figure()
    plt.plot(range(1, len(latencies) + 1), latencies, marker='o', linestyle='-')
    plt.title("Latency per Clip (Online Processing)")
    plt.xlabel("Clip Index")
    plt.ylabel("Latency (sec)")
    plt.grid(True)
    plt.show()

    # 3. Гистограмма распределения задержек
    plt.figure()
    plt.hist(latencies, bins=20, color='lightblue', edgecolor='black')
    plt.xlabel("Latency (sec)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Processing Latencies")
    plt.grid(True)
    plt.show()

    # 4. ROC-кривая
    fpr, tpr, thresholds = roc_curve(labels_list, probs_list)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # 5. Гистограмма распределения предсказанных вероятностей (для класса 1)
    plt.figure()
    plt.hist(probs_list, bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel("Predicted Probability (Class 1)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Probabilities")
    plt.grid(True)
    plt.show()

    # 6. Scatter Plot: Latency vs Predicted Probability
    plt.figure()
    plt.scatter(probs_list, latencies, color='purple', alpha=0.6)
    plt.xlabel("Predicted Probability (Class 1)")
    plt.ylabel("Latency (sec)")
    plt.title("Latency vs Predicted Probability")
    plt.grid(True)
    plt.show()
