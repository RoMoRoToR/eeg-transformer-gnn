import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import global_mean_pool

# Импортируем все компоненты новой модели и утилиты
from robust_hybrid_eeg_model import (
    RobustHybridEEGModel,
    EEGDataset,
    compute_adjacency,
    create_edge_index
)

###########################################
# Функция обучения и валидации
###########################################
def train_model(model, train_loader, val_loader, edge_index, device,
                num_epochs=50, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=5)

    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    for epoch in range(1, num_epochs+1):
        # ——— Тренировочный этап ———
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        for clips, labels in train_loader:
            clips  = clips.to(device)            # [B, C, T]
            labels = labels.to(device)

            # создаём batch_index для графа
            B, C, _ = clips.shape
            batch_idx = torch.arange(B, device=device)\
                              .unsqueeze(1)\
                              .repeat(1, C)\
                              .view(-1)

            optimizer.zero_grad()
            outputs = model(clips, edge_index, batch_idx)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * B
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += B

        epoch_loss = running_loss / total
        epoch_acc  = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # ——— Валидационный этап ———
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val   = 0

        with torch.no_grad():
            for clips, labels in val_loader:
                clips  = clips.to(device)
                labels = labels.to(device)

                B, C, _ = clips.shape
                batch_idx = torch.arange(B, device=device)\
                                  .unsqueeze(1)\
                                  .repeat(1, C)\
                                  .view(-1)

                outputs = model(clips, edge_index, batch_idx)
                loss    = criterion(outputs, labels)

                val_loss += loss.item() * B
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val   += B

        val_loss /= total_val
        val_acc  = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(f"Эпоха {epoch:02d}/{num_epochs}: "
              f"Train Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    return train_losses, train_accs, val_losses, val_accs

###########################################
# Функция для визуализации метрик
###########################################
def plot_curves(train_losses, train_accs, val_losses, val_accs):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Val   Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('Кривые потерь')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs,   label='Val   Acc')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.title('Кривые точности')
    plt.legend()

    plt.tight_layout()
    plt.show()

###########################################
# Основной блок
###########################################
if __name__ == "__main__":
    # Путь к данным и устройству
    data_root = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"  # <-- замените на ваш путь
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем датасет и разбиваем на train/val
    dataset = EEGDataset(data_root)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val   = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

    # Готовим граф: adjacency + edge_index
    n_channels = 19
    adj        = compute_adjacency(n_channels)
    edge_index = create_edge_index(adj, threshold=0.3).to(device)

    # Инициализируем модель
    model = RobustHybridEEGModel(
        n_channels= n_channels,
        n_samples = 500,
        feat_dim  = 16,
        gcn_hidden= 64,
        heads     = 8,
        num_classes=2,
        noise_sigma=0.1,
        dropout    =0.5
    ).to(device)

    # Тренируем
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, edge_index, device,
        num_epochs=50, lr=1e-3
    )

    # Визуализируем
    plot_curves(train_losses, train_accs, val_losses, val_accs)

    # Сохраняем веса
    torch.save(model.state_dict(), "robust_hybrid_eeg_model_final.pth")
    print("Обучение завершено и модель сохранена в robust_hybrid_eeg_model_final.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_ROOT = "preproc_clips_train"
    TEST_ROOT  = "preproc_clips_test"
    train_on_dirs(TRAIN_ROOT, TEST_ROOT, device)