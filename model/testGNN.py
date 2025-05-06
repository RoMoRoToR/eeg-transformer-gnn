# model/testGNN.py
"""
Adapted model for Alzheimer's diagnosis using streaming EEG data.
Some parts are adapted from the eeg-gnn-ssl repository.
"""

from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable

# Импортируем DCGRUCell из модуля cell (предполагается, что он есть в проекте)
from cell import DCGRUCell
import utils


# Функция для вычисления diffusion supports
def compute_diffusion_supports(A, max_diffusion_step=1):
    """
    Вычисляет diffusion supports для заданной матрицы A.
    A: torch.Tensor размера (num_nodes, num_nodes)
    Возвращает тензор размера ((max_diffusion_step+1)*num_nodes, num_nodes)
    """
    num_nodes = A.size(0)
    I = torch.eye(num_nodes, device=A.device)
    supports = [I]
    current_support = A
    for k in range(1, max_diffusion_step + 1):
        supports.append(current_support)
        current_support = torch.matmul(A, current_support)
    return torch.cat(supports, dim=0)  # (max_diffusion_step+1)*num_nodes x num_nodes


# ------------------ DCGRU Encoder ------------------
class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step, hid_dim, num_nodes, num_rnn_layers,
                 dcgru_activation='tanh', filter_type='laplacian', device=None):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self.num_nodes = num_nodes
        self._device = device
        encoding_cells = []
        # Первый слой имеет input_dim, затем все последующие — hid_dim
        encoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type))
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                DCGRUCell(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    nonlinearity=dcgru_activation,
                    filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state, supports):
        # inputs: (seq_len, batch, num_nodes, input_dim)
        seq_len = inputs.shape[0]
        batch_size = inputs.shape[1]
        # Приводим вход к форме (seq_len, batch, num_nodes * input_dim)
        inputs = torch.reshape(inputs, (seq_len, batch_size, -1))
        current_inputs = inputs
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_len):
                # Если hidden_state имеет форму (batch, num_nodes, rnn_units), выпрямляем его:
                if hidden_state.dim() == 3:
                    hidden_state = hidden_state.view(batch_size, -1)
                _, hidden_state = self.encoding_cells[i_layer](supports, current_inputs[t], hidden_state)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(self._device)
        output_hidden = torch.stack(output_hidden, dim=0).to(self._device)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            h = self.encoding_cells[i].init_hidden(batch_size)
            # Если скрытое состояние имеет форму (batch, num_nodes, rnn_units), выпрямляем его:
            if h.dim() == 3:
                h = h.view(batch_size, -1)
            init_states.append(h)
        return torch.stack(init_states, dim=0)



# ------------------ Alzheimer's Diagnosis Model ------------------
class AlzheimerDCGRNNClassifier(nn.Module):
    def __init__(self, input_dim, max_diffusion_step, num_nodes, num_rnn_layers,
                 rnn_units, num_classes, dcgru_activation='tanh',
                 filter_type='laplacian', dropout=0.5, device=None):
        super(AlzheimerDCGRNNClassifier, self).__init__()
        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.num_classes = num_classes

        self.encoder = DCRNNEncoder(
            input_dim=input_dim,
            max_diffusion_step=max_diffusion_step,
            hid_dim=rnn_units,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
            dcgru_activation=dcgru_activation,
            filter_type=filter_type,
            device=device
        )

        self.fc = nn.Linear(rnn_units, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input_seq, seq_lengths, supports):
        # input_seq: (batch_size, seq_len, num_nodes, input_dim)
        batch_size, seq_len, num_nodes, _ = input_seq.shape
        # Переставляем оси: (seq_len, batch_size, num_nodes, input_dim)
        encoder_inputs = input_seq.permute(1, 0, 2, 3)
        init_hidden = self.encoder.init_hidden(batch_size).to(self._device)
        _, final_hidden = self.encoder(encoder_inputs, init_hidden, supports)
        # Берем последнее скрытое состояние последнего слоя: (batch_size, num_nodes * rnn_units)
        last_hidden = final_hidden[-1]
        # Приводим к форме (batch_size, num_nodes, rnn_units)
        last_hidden = last_hidden.view(batch_size, num_nodes, self.rnn_units)
        last_hidden = last_hidden.to(self._device)
        # Макс-пулинг по узлам
        pooled, _ = torch.max(last_hidden, dim=1)
        out = self.fc(self.relu(self.dropout(pooled)))
        return out



# Пример использования модели для проверки forward pass
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Alzheimer's Diagnosis Model with Streaming EEG")
    parser.add_argument("--input_dim", type=int, default=7, help="Dimension of node features")
    parser.add_argument("--max_diffusion_step", type=int, default=1, help="Max diffusion step for DCGRU")
    parser.add_argument("--num_nodes", type=int, default=19, help="Number of EEG channels (nodes)")
    parser.add_argument("--num_rnn_layers", type=int, default=2, help="Number of DCGRU layers")
    parser.add_argument("--rnn_units", type=int, default=64, help="Number of hidden units in DCGRU")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создаем фиксированную матрицу смежности (например, кольцевую топологию)
    num_nodes = args.num_nodes
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        A[i, j] = 1
        A[j, i] = 1
    A = torch.tensor(A, dtype=torch.float, device=device)
    # Вычисляем diffusion supports: размер ((max_diffusion_step+1)*num_nodes, num_nodes)
    # supports = compute_diffusion_supports(A, max_diffusion_step=args.max_diffusion_step).to(device)

    # Вычисляем supports как тензор
    supports_tensor = compute_diffusion_supports(A, max_diffusion_step=args.max_diffusion_step).to(device)
    # Разбиваем supports на список матриц
    num_nodes = args.num_nodes
    max_diffusion_step = args.max_diffusion_step
    supports = [supports_tensor[i * num_nodes:(i + 1) * num_nodes] for i in range(max_diffusion_step + 1)]

    # Пример входных данных:
    batch_size = 8
    seq_len = 10
    input_tensor = torch.randn(batch_size, seq_len, num_nodes, args.input_dim).to(device)
    seq_lengths = torch.full((batch_size,), seq_len, dtype=torch.long).to(device)

    model = AlzheimerDCGRNNClassifier(
        input_dim=args.input_dim,
        max_diffusion_step=args.max_diffusion_step,
        num_nodes=args.num_nodes,
        num_rnn_layers=args.num_rnn_layers,
        rnn_units=args.rnn_units,
        num_classes=args.num_classes,
        dropout=args.dropout,
        device=device
    ).to(device)

    logits = model(input_tensor, seq_lengths, supports)
    print("Logits shape:", logits.shape)  # Ожидаем (batch_size, num_classes)

    # Здесь можно добавить цикл обучения, DataLoader для вашего датасета, и т.д.
