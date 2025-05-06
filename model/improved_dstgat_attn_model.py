import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch_geometric.nn import GATConv, global_mean_pool
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
        # x: [batch*n_channels, input_dim, n_samples]
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
        # x: [B, channels]
        s = x.mean(dim=0, keepdim=True)  # [1, channels]
        s = self.fc1(s)
        s = F.gelu(s)
        s = self.fc2(s)
        s = torch.sigmoid(s)
        return x * s

###########################################
# Функции для работы с графами
###########################################

def create_edge_index(adj_matrix, threshold=0.1):
    src, dst = np.where((adj_matrix > threshold) & (np.eye(adj_matrix.shape[0]) == 0))
    edge_index = np.vstack((src, dst))
    return torch.tensor(edge_index, dtype=torch.long)

def compute_adjacency(n_channels):
    coords = np.random.rand(n_channels, 2)  # При наличии реальных координат заменить генерацию
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    eps = 1e-5
    adj = 1.0 / (dist + eps)
    np.fill_diagonal(adj, 0)
    adj = adj / np.max(adj)
    return adj

###########################################
# Улучшенная модель DSTGAT_Attn с дополнительными методами оптимизации
###########################################

class Improved_DSTGAT_Attn(nn.Module):
    def __init__(self, n_channels, kernel_sizes=[3, 5, 7],
                 temporal_channels=16, gcn_channels=32,
                 num_classes=2, dropout_rate=0.5, gat_heads=4, attn_hidden=32):
        """
        n_channels: число электродов (узлов графа)
        """
        super(Improved_DSTGAT_Attn, self).__init__()

        # Многомасштабная временная свёртка
        self.temp_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for k in kernel_sizes:
            conv = nn.Conv1d(in_channels=1, out_channels=temporal_channels,
                             kernel_size=k, padding=k // 2)
            bn = nn.BatchNorm1d(temporal_channels)
            attn = TemporalAttention(input_dim=temporal_channels, hidden_dim=attn_hidden)
            self.temp_convs.append(conv)
            self.bns.append(bn)
            self.attentions.append(attn)

        self.dropout = nn.Dropout(dropout_rate)
        self.se = SEBlock(temporal_channels * len(kernel_sizes), reduction=4)

        # Графовые слои с residual connections
        self.gat1 = GATConv(temporal_channels * len(kernel_sizes), gcn_channels,
                            heads=gat_heads, concat=False)
        self.gat1_bn = nn.BatchNorm1d(gcn_channels)
        self.skip1 = nn.Linear(temporal_channels * len(kernel_sizes), gcn_channels)

        self.gat2 = GATConv(gcn_channels, gcn_channels, heads=gat_heads, concat=False)
        self.gat2_bn = nn.BatchNorm1d(gcn_channels)

        self.fc = nn.Linear(gcn_channels, num_classes)

        self.n_channels = n_channels
        self.kernel_sizes = kernel_sizes

    def forward(self, x, edge_index):
        # x: [batch_size, n_channels, n_samples]
        batch_size, n_channels, n_samples = x.shape
        x = x.view(batch_size * n_channels, 1, n_samples)

        branch_features = []
        for conv, bn, attn in zip(self.temp_convs, self.bns, self.attentions):
            out = conv(x)              # [B*n_channels, temporal_channels, n_samples]
            out = bn(out)
            out = F.gelu(out)
            out = self.dropout(out)
            attn_out = attn(out)       # [B*n_channels, temporal_channels]
            branch_features.append(attn_out)

        x = torch.cat(branch_features, dim=1)  # [B*n_channels, temporal_channels * num_branches]
        x = self.se(x)

        x = x.view(batch_size, n_channels, -1).view(batch_size * n_channels, -1)

        # Подготовка edge_index для батча: смещение индексов для каждого примера
        new_edge_index = []
        for i in range(batch_size):
            new_edge_index.append(edge_index + i * n_channels)
        new_edge_index = torch.cat(new_edge_index, dim=1).to(x.device)
        batch_vector = torch.arange(batch_size, device=x.device).unsqueeze(1).repeat(1, n_channels).view(-1)

        # Графовые слои с residual connections
        x_in = x
        x = self.gat1(x, new_edge_index)
        x = self.gat1_bn(x)
        x_skip = self.skip1(x_in)
        x = F.gelu(x + x_skip)
        x = self.dropout(x)

        x_in2 = x
        x = self.gat2(x, new_edge_index)
        x = self.gat2_bn(x)
        x = F.gelu(x + x_in2)
        x = self.dropout(x)

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
        clip = data['clip']  # [n_channels, n_samples]
        label = data['label']
        clip = torch.tensor(clip, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return clip, label

###########################################
# Обучение и оценка модели с выводом графиков
###########################################

def train_model(model, train_loader, test_loader, device, num_epochs=50, lr=1e-3):
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
            # Градиентный клиппинг
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

        # Тестирование
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
        print(f"Эпоха {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

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

    # Вычисляем матрицу смежности для графовых слоев
    n_channels = 19
    adj = compute_adjacency(n_channels)
    edge_index = create_edge_index(adj, threshold=0.3).to(device)

    # Инициализируем модель
    model = Improved_DSTGAT_Attn(n_channels=n_channels, kernel_sizes=[3,5,7],
                                  temporal_channels=16, gcn_channels=32, num_classes=2,
                                  dropout_rate=0.5, gat_heads=4, attn_hidden=32)
    model.to(device)

    train_losses, train_accs, test_losses, test_accs = train_model(model, train_loader, test_loader, device, num_epochs=50, lr=1e-3)
    plot_curves(train_losses, train_accs, test_losses, test_accs)

    torch.save(model.state_dict(), "improved_dstgcn_model_final.pth")
    print("Обучение завершено, модель сохранена.")
