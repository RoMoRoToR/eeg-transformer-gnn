import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from torch.utils.data import Dataset

# Импортируйте вашу модель и утилиты
from RobustHybridEEGModelV2 import RobustHybridEEGModelV2, compute_adjacency, create_edge_index

class EEGDatasetPlain(Dataset):
    """
    Тот же датасет, но без аугментаций.
    Возвращает clip: Tensor[C, T], label: int
    """
    def __init__(self, root_dir):
        self.paths = []
        for subj in os.listdir(root_dir):
            p = os.path.join(root_dir, subj)
            if os.path.isdir(p):
                for f in os.listdir(p):
                    if f.endswith('.npz'):
                        self.paths.append(os.path.join(p, f))
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        clip  = torch.tensor(data['clip'], dtype=torch.float32)  # [C, T]
        label = int(data['label'])
        return clip, label

def evaluate_with_noise(model, dataset, edge_index, device, noise_levels):
    accs = []
    aucs = []
    model.eval()
    for sigma in noise_levels:
        preds = []
        labs  = []
        probs = []
        with torch.no_grad():
            for clip, label in dataset:
                # clip: [C, T] -> [1, C, T]
                x = clip.unsqueeze(0).to(device)
                # добавляем гауссов шум
                if sigma > 0:
                    x = x + torch.randn_like(x) * sigma
                # batch_idx для графа
                B, C, T = x.shape
                batch_idx = torch.arange(B, device=device)\
                                 .unsqueeze(1).repeat(1, C).view(-1)
                # предсказание
                logits = model(x, edge_index, batch_idx)
                p = F.softmax(logits, dim=1)[0,1].item()
                pred = logits.argmax(dim=1)[0].item()

                probs.append(p)
                preds.append(pred)
                labs.append(label)

        acc = accuracy_score(labs, preds) * 100
        fpr, tpr, _ = roc_curve(labs, probs)
        roc_auc = auc(fpr, tpr) * 100

        print(f"σ={sigma:.3f} → Acc={acc:.2f}%, AUC={roc_auc:.2f}%")
        accs.append(acc)
        aucs.append(roc_auc)

    return accs, aucs

if __name__ == "__main__":
    # Параметры
    data_root   = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"
    model_path  = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/model/best_model_v2.pth"
    n_channels  = 19
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    model = RobustHybridEEGModelV2(
        n_channels=n_channels,
        n_samples=500,
        feat_dim=16,
        gcn_hidden=64,
        heads=8,
        num_classes=2,
        noise_sigma=0.0,  # отключаем шум внутри модели
        dropout=0.0       # отключаем дропаут
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Подготовка графа
    adj        = compute_adjacency(n_channels)
    edge_index = create_edge_index(adj, threshold=0.3).to(device)

    # Датасет
    dataset = EEGDatasetPlain(data_root)

    # Уровни шума
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Оценка
    accs, aucs = evaluate_with_noise(model, dataset, edge_index, device, noise_levels)

    # Визуализация
    plt.figure(figsize=(8,5))
    plt.plot(noise_levels, accs, '-o', label='Accuracy (%)')
    plt.plot(noise_levels, aucs, '-s', label='AUC ×100 (%)')
    plt.xlabel('Noise σ')
    plt.ylabel('Value (%)')
    plt.title('Stability Curve of RobustHybridEEGModelV2')
    plt.ylim(0,100)
    plt.grid(True)
    plt.legend()
    plt.show()
