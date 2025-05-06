import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from model.hybrid_eegnet_gat_model_final import create_edge_index, compute_adjacency


###########################################
# 1D-модули на основе EEGNet для извлечения временных признаков
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
        return self.pointwise(x)

class EEGFeatureExtractor1D(nn.Module):
    """
    Извлекает временные признаки для каждого электрода.
    Входной формат: [B*n_channels, 1, n_samples]
    Выход: [B*n_channels, feat_dim]
    """
    def __init__(self, temporal_filters=16, output_dim=16, kernel_size=25, pooling_kernel=4, dropout_rate=0.5):
        super(EEGFeatureExtractor1D, self).__init__()
        self.conv1 = nn.Conv1d(1, temporal_filters, kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(temporal_filters)
        self.elu = nn.ELU()
        self.depthwise = DepthwiseConv1d(temporal_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(temporal_filters)
        self.separable = SeparableConv1d(temporal_filters, output_dim, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(output_dim)
        self.pool = nn.AvgPool1d(pooling_kernel)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        x = self.conv1(x)             # [B*n_channels, temporal_filters, n_samples]
        x = self.bn1(x)
        x = self.elu(x)
        x = self.depthwise(x)         # [B*n_channels, temporal_filters, n_samples]
        x = self.bn2(x)
        x = self.elu(x)
        x = self.separable(x)         # [B*n_channels, output_dim, n_samples]
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = F.adaptive_avg_pool1d(x, 1)  # [B*n_channels, output_dim, 1]
        return x.squeeze(-1)           # [B*n_channels, output_dim]

###########################################
# Графовый блок (на основе GATConv) с residual-подключениями
###########################################

class GraphBlock(nn.Module):
    def __init__(self, in_features, gcn_channels=32, dropout_rate=0.5, gat_heads=4):
        super(GraphBlock, self).__init__()
        self.gat1 = GATConv(in_features, gcn_channels, heads=gat_heads, concat=False)
        self.gat1_bn = nn.BatchNorm1d(gcn_channels)
        self.skip1 = nn.Linear(in_features, gcn_channels)
        self.gat2 = GATConv(gcn_channels, gcn_channels, heads=gat_heads, concat=False)
        self.gat2_bn = nn.BatchNorm1d(gcn_channels)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x, edge_index, batch_vector):
        x_in = x
        x = self.gat1(x, edge_index)
        x = self.gat1_bn(x)
        x_skip = self.skip1(x_in)
        x = F.gelu(x + x_skip)
        x = self.dropout(x)
        x_in2 = x
        x = self.gat2(x, edge_index)
        x = self.gat2_bn(x)
        x = F.gelu(x + x_in2)
        x = self.dropout(x)
        x = global_mean_pool(x, batch_vector)
        return x

###########################################
# Гибридная модель Hybrid_EEG_GNN (объединение EEGFeatureExtractor1D и GraphBlock)
###########################################

class Hybrid_EEG_GNN(nn.Module):
    def __init__(self, n_channels, n_samples, dropout_rate=0.5, gcn_channels=32, gat_heads=4, num_classes=2):
        super(Hybrid_EEG_GNN, self).__init__()
        self.n_channels = n_channels
        # EEG-блок для извлечения временных признаков
        self.feature_extractor = EEGFeatureExtractor1D(temporal_filters=16, output_dim=16,
                                                       kernel_size=25, pooling_kernel=4, dropout_rate=dropout_rate)
        # Графовый блок
        self.graph_block = GraphBlock(in_features=16, gcn_channels=gcn_channels,
                                       dropout_rate=dropout_rate, gat_heads=gat_heads)
        # Классификатор
        self.fc = nn.Linear(gcn_channels, num_classes)
    def forward(self, x, edge_index):
        # x: [B, 1, n_channels, n_samples]
        B = x.size(0)
        # Извлечение временных признаков: преобразуем x в [B*n_channels, 1, n_samples]
        features = self.feature_extractor(x.view(B * self.n_channels, 1, -1))  # -> [B*n_channels, feat_dim]
        # Восстанавливаем форму [B, n_channels, feat_dim] и затем в [B*n_channels, feat_dim]
        features = features.view(B, self.n_channels, -1).view(B * self.n_channels, -1)
        # Создаем batch_vector: для каждого узла указываем принадлежность к батчу
        batch_vector = torch.arange(B, device=x.device).unsqueeze(1).repeat(1, self.n_channels).view(-1)
        # Графовый блок
        graph_out = self.graph_block(features, edge_index, batch_vector)
        logits = self.fc(graph_out)
        return logits

###########################################
# Dataset для EEG данных
###########################################

class EEGDataset(Dataset):
    def __init__(self, root_dir):
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
        clip = data['clip']    # [n_channels, n_samples]
        label = data['label']
        # Для модели Hybrid_EEG_GNN входной формат: [1, n_channels, n_samples]
        clip = clip[np.newaxis, ...]
        clip = torch.tensor(clip, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return clip, label

###########################################
# Онлайн-тестирование модели Hybrid_EEG_GNN с выводом графиков
###########################################

if __name__ == "__main__":
    data_root = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"  # Укажите путь к данным
    # Обратите внимание: замените model_path на корректный путь к сохранённой модели Hybrid_EEG_GNN,
    # например "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/model/hybrid_eeg_gnn_model_final.pth"
    model_path = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/model/hybrid_eeg_gnn_model_final.pth"
    n_channels = 19
    n_samples = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Инициализация модели
    model = Hybrid_EEG_GNN(n_channels=n_channels, n_samples=n_samples, dropout_rate=0.5,
                           gcn_channels=32, gat_heads=4, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Вычисление матрицы смежности и формирование edge_index
    adj = compute_adjacency(n_channels)
    edge_index = create_edge_index(adj, threshold=0.3).to(device)

    # Загрузка датасета
    dataset = EEGDataset(data_root)

    print("Начало онлайн-тестирования модели Hybrid_EEG_GNN...")
    preds_list = []
    labels_list = []
    latencies = []
    probs_list = []  # Для ROC-кривой

    for i in range(len(dataset)):
        clip, label = dataset[i]
        # Добавляем размерность батча: [1, 1, n_channels, n_samples]
        clip = clip.unsqueeze(0).to(device)
        label = label.to(device)
        start_time = time.time()
        with torch.no_grad():
            output = model(clip, edge_index)
        latency = time.time() - start_time
        latencies.append(latency)
        # Для CrossEntropyLoss output имеет форму [1, num_classes]
        prob = F.softmax(output, dim=1)[0, 1].item()
        probs_list.append(prob)
        _, pred = torch.max(output, dim=1)
        preds_list.append(pred.item())
        labels_list.append(label.item())
        print(f"Клип {i+1}/{len(dataset)}: Предсказано {pred.item()}, Истинное {label.item()}, Время: {latency:.4f} сек.")

    accuracy = 100.0 * sum(int(p == l) for p, l in zip(preds_list, labels_list)) / len(labels_list)
    avg_latency = np.mean(latencies)
    print(f"\nОнлайн-тестирование завершено. Общая точность: {accuracy:.2f}%")
    print(f"Среднее время обработки клипа: {avg_latency:.4f} сек.")

    # =======================
    # Построение графиков для оценки модели
    # =======================

    # 1. Конфузионная матрица и отчет по классификации
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

    # 2. График задержек (latency) по клипам
    plt.figure()
    plt.plot(range(1, len(latencies)+1), latencies, marker='o', linestyle='-')
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
    plt.plot(fpr, tpr, lw=2, color='darkorange', label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], lw=2, linestyle='--', color='navy')
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

    # 6. Scatter plot: задержка обработки vs предсказанная вероятность
    plt.figure()
    plt.scatter(probs_list, latencies, color='purple', alpha=0.6)
    plt.xlabel("Predicted Probability (Class 1)")
    plt.ylabel("Latency (sec)")
    plt.title("Latency vs Predicted Probability")
    plt.grid(True)
    plt.show()
