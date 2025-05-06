import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from optimized_dstgat_attn_model_final import (
    Optimized_DSTGAT_Attn,
    EEGDataset,
    compute_adjacency,
    create_edge_index
)

def evaluate_with_noise(model, dataset, edge_idx, device, noise_std):
    preds, labels, probs = [], [], []
    for idx in range(len(dataset)):
        clip, label = dataset[idx]
        clip = clip.unsqueeze(0).to(device)  # [1, C, T]
        # add Gaussian noise
        clip = clip + noise_std * torch.randn_like(clip)
        with torch.no_grad():
            out = model(clip, edge_idx)
        prob = F.softmax(out, dim=1)[0,1].item()
        pred = int(out.argmax(dim=1).item())
        preds.append(pred)
        labels.append(int(label))
        probs.append(prob)
    # metrics
    acc = np.mean([p==l for p,l in zip(preds,labels)])
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    return acc, roc_auc

if __name__ == "__main__":
    # paths and params
    data_root  = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"
    model_path = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/model/optimized_dstgat_attn_model_final.pth"
    n_channels = 19
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data and model
    dataset = EEGDataset(data_root)
    adj     = compute_adjacency(n_channels)
    edge_idx= create_edge_index(adj, threshold=0.3).to(device)

    model = Optimized_DSTGAT_Attn(
        n_channels=n_channels,
        eeg_feat_dim=16,
        gcn_channels=32,
        num_classes=2,
        dropout_rate=0.5,
        gat_heads=4
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # noise levels to test
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    accuracies = []
    aucs = []

    for σ in noise_levels:
        acc, roc_auc = evaluate_with_noise(model, dataset, edge_idx, device, noise_std=σ)
        accuracies.append(acc)
        aucs.append(roc_auc)
        print(f"σ={σ:.3f} → Accuracy={acc*100:.2f}%, AUC={roc_auc:.3f}")

    # plot stability curves
    plt.figure(figsize=(8,4))
    plt.plot(noise_levels, np.array(accuracies)*100, marker='o', label='Accuracy (%)')
    plt.plot(noise_levels, np.array(aucs)*100,      marker='s', label='AUC ×100')
    plt.xlabel('Noise σ')
    plt.title('Stability Curve: Accuracy & AUC vs Noise Level')
    plt.grid(True)
    plt.legend()
    plt.show()
