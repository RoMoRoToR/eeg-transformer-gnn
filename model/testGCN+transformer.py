import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv, global_mean_pool
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt

###########################################
# Модуль внимания для временной агрегации
###########################################

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        """
        input_dim: число входных признаков (например, temporal_channels)
        hidden_dim: размер скрытого пространства для attention
        """
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        x: тензор размера [batch*n_channels, input_dim, n_samples]
        Возвращает: [batch*n_channels, input_dim] – взвешенная сумма по временной оси.
        """
        attn_scores = self.attn(x)  # [B*n_channels, 1, n_samples]
        attn_weights = F.softmax(attn_scores, dim=2)  # нормировка по временной оси
        weighted = (x * attn_weights).sum(dim=2)  # [B*n_channels, input_dim]
        return weighted

###########################################
# Squeeze-and-Excitation блок
###########################################

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        """
        channels: число входных каналов
        reduction: коэффициент сжатия
        """
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        """
        x: [B, channels]
        """
        # Вычисляем статистику по каналам (глобальное среднее)
        s = x.mean(dim=0, keepdim=True)  # [1, channels]
        s = self.fc1(s)
        s = F.gelu(s)
        s = self.fc2(s)
        s = torch.sigmoid(s)
        return x * s

###########################################
# Модуль Transformer для обработки временной оси
###########################################

class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=1, dropout=0.1):
        """
        d_model: размерность входных признаков (должна совпадать с числом фильтров свёртки)
        nhead: число голов внимания
        num_layers: число слоёв трансформера
        dropout: dropout внутри трансформера
        """
        super(TemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: тензор размера [B, d_model, n_samples]
        Приводим к формату [n_samples, B, d_model] для работы Transformer, затем возвращаем обратно.
        """
        x = x.permute(2, 0, 1)  # [n_samples, B, d_model]
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # [B, d_model, n_samples]
        return x

###########################################
# Функции для работы с графами
###########################################

def create_edge_index(adj_matrix, threshold=0.1):
    """
    Создает edge_index из матрицы смежности.
    adj_matrix: np.array размера (n_channels, n_channels)
    threshold: порог для отбора ребер.
    Возвращает edge_index в формате torch.LongTensor (2, num_edges)
    """
    src, dst = np.where((adj_matrix > threshold) & (np.eye(adj_matrix.shape[0]) == 0))
    edge_index = np.vstack((src, dst))
    return torch.tensor(edge_index, dtype=torch.long)

def compute_adjacency(n_channels):
    """
    Вычисляет матрицу смежности на основе случайных координат электродов.
    При наличии реальных координат (схема 10–20) замените генерацию.
    """
    coords = np.random.rand(n_channels, 2)  # Используйте реальные координаты при наличии
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    eps = 1e-5
    adj = 1.0 / (dist + eps)
    np.fill_diagonal(adj, 0)
    adj = adj / np.max(adj)
    return adj

###########################################
# Улучшенная модель с residual connections, SE-блоком и Transformer
###########################################

class Improved_DSTGAT_Attn(nn.Module):
    def __init__(self, n_channels, kernel_sizes=[3, 5, 7],
                 temporal_channels=16, gcn_channels=32,
                 num_classes=2, dropout_rate=0.5, gat_heads=4, attn_hidden=32):
        """
        n_channels: число электродов (узлов графа)
        kernel_sizes: список размеров ядер для многошкальной временной свёртки
        temporal_channels: число фильтров для каждой свёртки
        gcn_channels: число выходных признаков после GAT
        num_classes: число классов (например, 2 – Альцгеймер и контроль)
        dropout_rate: вероятность dropout
        gat_heads: число "голов" в GATConv
        attn_hidden: размер скрытого пространства в TemporalAttention
        """
        super(Improved_DSTGAT_Attn, self).__init__()

        # Многошкальная временная свёртка
        self.temp_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.transformers = nn.ModuleList()  # трансформер для каждой ветки
        self.attentions = nn.ModuleList()
        for k in kernel_sizes:
            conv = nn.Conv1d(in_channels=1, out_channels=temporal_channels,
                             kernel_size=k, padding=k // 2)
            bn = nn.BatchNorm1d(temporal_channels)
            # Трансформер для обработки временной последовательности после свёртки
            transformer = TemporalTransformer(d_model=temporal_channels, nhead=4, num_layers=1, dropout=dropout_rate)
            attn = TemporalAttention(input_dim=temporal_channels, hidden_dim=attn_hidden)
            self.temp_convs.append(conv)
            self.bns.append(bn)
            self.transformers.append(transformer)
            self.attentions.append(attn)

        self.dropout = nn.Dropout(dropout_rate)

        # SE-блок для калибровки мультишкальных признаков
        self.se = SEBlock(temporal_channels * len(kernel_sizes), reduction=4)

        # Первый GAT-слой с residual connection
        self.gat1 = GATConv(temporal_channels * len(kernel_sizes), gcn_channels,
                            heads=gat_heads, concat=False)
        self.gat1_bn = nn.BatchNorm1d(gcn_channels)
        # Проекция для skip connection (если размерности не совпадают)
        self.skip1 = nn.Linear(temporal_channels * len(kernel_sizes), gcn_channels)

        # Второй GAT-слой с residual connection
        self.gat2 = GATConv(gcn_channels, gcn_channels, heads=gat_heads, concat=False)
        self.gat2_bn = nn.BatchNorm1d(gcn_channels)
        # Для второго слоя skip connection используется identity (размерности совпадают)

        # Финальный классификатор
        self.fc = nn.Linear(gcn_channels, num_classes)

        self.n_channels = n_channels
        self.kernel_sizes = kernel_sizes

    def forward(self, x, edge_index):
        """
        x: [batch_size, n_channels, n_samples]
        edge_index: [2, num_edges] базовая структура графа (одинакова для всех примеров)
        """
        batch_size, n_channels, n_samples = x.shape
        # Объединяем batch и n_channels для параллельной обработки
        x = x.view(batch_size * n_channels, 1, n_samples)  # [B*n_channels, 1, n_samples]

        branch_features = []
        # Для каждой параллельной ветки: свёртка -> BN -> GELU -> Dropout -> Transformer -> Temporal Attention
        for conv, bn, transformer, attn in zip(self.temp_convs, self.bns, self.transformers, self.attentions):
            out = conv(x)                     # [B*n_channels, temporal_channels, n_samples]
            out = bn(out)
            out = F.gelu(out)
            out = self.dropout(out)
            out = transformer(out)            # трансформер обрабатывает временную последовательность
            attn_out = attn(out)              # агрегируем признаки по временной оси: [B*n_channels, temporal_channels]
            branch_features.append(attn_out)

        # Конкатенация признаков от всех веток
        x = torch.cat(branch_features, dim=1)  # [B*n_channels, temporal_channels * num_branches]

        # Применяем SE-блок для переобучения каналов
        x = self.se(x)

        # Восстанавливаем размерность батча для графовой обработки
        x = x.view(batch_size, n_channels, -1)
        x = x.view(batch_size * n_channels, -1)

        # Подготовка edge_index для батча: смещение индексов для каждого примера
        new_edge_index = []
        for i in range(batch_size):
            new_edge_index.append(edge_index + i * n_channels)
        new_edge_index = torch.cat(new_edge_index, dim=1).to(x.device)

        # Вектор принадлежности узлов для глобального пуллинга
        batch_vector = torch.arange(batch_size, device=x.device).unsqueeze(1).repeat(1, n_channels).view(-1)

        # Первый GAT-слой с residual connection
        x_in = x  # сохраняем для skip connection
        x = self.gat1(x, new_edge_index)
        x = self.gat1_bn(x)
        x_skip = self.skip1(x_in)
        x = F.gelu(x + x_skip)
        x = self.dropout(x)

        # Второй GAT-слой с residual connection
        x_in2 = x
        x = self.gat2(x, new_edge_index)
        x = self.gat2_bn(x)
        x = F.gelu(x + x_in2)
        x = self.dropout(x)

        # Глобальный пуллинг по узлам каждого графа
        x = global_mean_pool(x, batch_vector)  # [batch_size, gcn_channels]

        # Классификация
        logits = self.fc(x)
        return logits

###########################################
# Пользовательский Dataset (прием папки с данными)
###########################################

class EEGDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: путь к каталогу с предобработанными клипами.
        Ожидаемая структура:
          root_dir/
            subject1/
              subject1_clip_0.npz
              subject1_clip_1.npz
              ...
            subject2/
              subject2_clip_0.npz
              ...
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
        return clip, label

###########################################
# Функции обучения и оценки
###########################################

def train_epoch(model, dataloader, optimizer, criterion, device, edge_index):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for clips, labels in dataloader:
        clips = clips.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(clips, edge_index.to(device))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * clips.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, dataloader, criterion, device, edge_index):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for clips, labels in dataloader:
            clips = clips.to(device)
            labels = labels.to(device)
            outputs = model(clips, edge_index.to(device))
            loss = criterion(outputs, labels)
            running_loss += loss.item() * clips.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

###########################################
# Основной блок: обучение модели
###########################################

if __name__ == "__main__":
    data_root = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"  # Замени
    n_channels = 19
    num_classes = 2
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-3

    dataset = EEGDataset(data_root)

    num_total = len(dataset)
    num_train = int(0.8 * num_total)
    num_val = num_total - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

    # Подсчет весов для сбалансированного обучения
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    class_sample_count = np.array([train_labels.count(cls_id) for cls_id in range(num_classes)])
    weight_per_class = 1.0 / class_sample_count
    samples_weight = np.array([weight_per_class[cls_id] for cls_id in train_labels])
    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    adj = compute_adjacency(n_channels)
    edge_index = create_edge_index(adj, threshold=0.3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Improved_DSTGAT_Attn(n_channels=n_channels, kernel_sizes=[3, 5, 7],
                                 temporal_channels=16, gcn_channels=32, num_classes=num_classes,
                                 dropout_rate=0.5, gat_heads=4, attn_hidden=32)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, edge_index)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, edge_index)
        scheduler.step()
        print(f"Эпоха {epoch + 1}/{num_epochs}:")
        print(f"  Обучение - Потери: {train_loss:.4f}  Точность: {train_acc:.4f}")
        print(f"  Валидация - Потери: {val_loss:.4f}  Точность: {val_acc:.4f}")

    torch.save(model.state_dict(), "improved_dstgcn_model_final.pth")
    print("Обучение завершено, модель сохранена.")
