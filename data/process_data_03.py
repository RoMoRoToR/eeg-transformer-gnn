import os
import numpy as np
import pandas as pd
import torch
import mne
import random
from torch_geometric.data import Data

# --------------------------------------------------------------------------------
# 1) Настройки
# --------------------------------------------------------------------------------

BIDS_ROOT = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/ds004504"
OUT_FILE = "my_eeg_dataset.pt"  # Итоговый файл с List[Data]
PARTICIPANTS_TSV = "participants.tsv"

# Параметры фильтрации EEG
L_FREQ, H_FREQ = 0.5, 45.0

# Настройки ICA
ICA_NCOMP = 19  # Число компонентов ICA
DO_ICA = True  # Флаг применения ICA

# Параметры сегментации (скользящее окно)
WINDOW_SEC = 10.0  # Длительность окна в секундах
STEP_SEC = 2.0  # Шаг окна в секундах

# Маппинг групп: оставляем только A (Alzheimer) и C (Control)
group_map = {
    'A': 1,  # Alzheimer
    'C': 0  # Control
}


# --------------------------------------------------------------------------------
# 2) Функции предобработки и построения графа
# --------------------------------------------------------------------------------

def build_edge_index(num_channels=19, connectivity_matrix=None, threshold=0.5):
    """
    Строит графовую топологию.
    Если connectivity_matrix (например, корреляционная матрица) передана, то для каждой пары каналов,
    у которых значение превышает порог, создается ребро.
    Если матрица не передана, используется фиксированная (например, кольцевая) топология.
    """
    if connectivity_matrix is None:
        edge_indices = []
        for i in range(num_channels):
            j = (i + 1) % num_channels
            edge_indices.append([i, j])
            edge_indices.append([j, i])
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return edge_index
    else:
        edge_indices = []
        for i in range(num_channels):
            for j in range(num_channels):
                if i != j and connectivity_matrix[i, j] > threshold:
                    edge_indices.append([i, j])
        if not edge_indices:
            # Если не найдено ни одного ребра, добавляем self-loop для каждого узла
            edge_indices = [[i, i] for i in range(num_channels)]
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return edge_index


def process_one_subject(set_fname, l_freq=1., h_freq=40., do_ica=False, ica_ncomp=15):
    """
    Загружает .set файл EEG, применяет фильтрацию, устанавливает average reference,
    и опционально применяет ICA для удаления артефактов.
    Возвращает объект Raw (MNE).
    """
    print(f"Loading: {set_fname}")
    raw = mne.io.read_raw_eeglab(set_fname, preload=True)
    print(f"Filtering {l_freq}–{h_freq} Hz")
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw.set_eeg_reference('average', projection=False)
    if do_ica:
        print(f"Applying ICA with {ica_ncomp} components.")
        ica = mne.preprocessing.ICA(n_components=ica_ncomp, random_state=42)
        ica.fit(raw)
        raw = ica.apply(raw)
    return raw


def extract_channel_features(seg, sfreq):
    """
    Вычисляет временные и спектральные признаки для каждого канала в сегменте.
    Параметры:
      seg: массив размерности (C, T) – где C – число каналов, T – число отсчетов
      sfreq: частота дискретизации EEG
    Вычисляем следующие признаки для каждого канала:
      - Среднее значение (mean)
      - Стандартное отклонение (std)
      - Мощность в диапазоне delta (1–4 Гц)
      - Мощность в диапазоне theta (4–8 Гц)
      - Мощность в диапазоне alpha (8–12 Гц)
      - Мощность в диапазоне beta (12–30 Гц)
      - Мощность в диапазоне gamma (30–45 Гц)
    Для удобства обучения признаки масштабируются (умножаются на 1e5).
    Опционально можно добавить z-score нормализацию.
    Возвращает массив размерности (C, 7).
    """
    n_channels, n_times = seg.shape
    seg_mean = seg.mean(axis=1)
    seg_std = seg.std(axis=1)

    psds, freqs = mne.time_frequency.psd_array_welch(seg, sfreq=sfreq, fmin=0.5, fmax=45, n_fft=256)

    delta_mask = (freqs >= 1) & (freqs < 4)
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs < 12)
    beta_mask = (freqs >= 12) & (freqs < 30)
    gamma_mask = (freqs >= 30) & (freqs <= 45)

    delta_power = psds[:, delta_mask].mean(axis=1)
    theta_power = psds[:, theta_mask].mean(axis=1)
    alpha_power = psds[:, alpha_mask].mean(axis=1)
    beta_power = psds[:, beta_mask].mean(axis=1)
    gamma_power = psds[:, gamma_mask].mean(axis=1)

    features = np.stack([seg_mean, seg_std, delta_power, theta_power, alpha_power, beta_power, gamma_power], axis=1)

    # Масштабирование: умножение на 1e5 для приведения значений к удобному диапазону.
    features = features * 1e5

    # Опционально: можно применить z-score нормализацию (раскомментируйте, если требуется)
    # features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    return features


def cumulative_graph_from_raw(raw, win_sec=5.0, step_sec=2.0, label=0, threshold=0.5):
    """
    Формирует накопительный (cumulative) граф для одного сеанса EEG.
    Для каждого окна:
      - Извлекаются признаки по каналам.
      - Вычисляется матрица функциональной связности (здесь используется корреляция).
      - Строится локальный граф с использованием порога.
    Затем:
      - Все узлы из всех окон объединяются в один большой граф.
      - Для корректировки индексов узлов производится смещение.
      - Добавляются временные ребра, связывающие узлы одного канала из соседних окон.
    Возвращает объект Data (PyTorch Geometric) с признаками x, ребрами edge_index и меткой y.
    """
    sfreq = raw.info['sfreq']
    data_arr = raw.get_data()  # (C, N)
    n_channels = data_arr.shape[0]
    win_samps = int(win_sec * sfreq)
    step_samps = int(step_sec * sfreq)
    n_times = data_arr.shape[1]

    node_features_list = []
    edge_indices_list = []
    window_indices = []  # Для хранения смещений узлов для каждого окна
    cumulative_offset = 0
    window_count = 0

    while (window_count * step_samps + win_samps) <= n_times:
        start = window_count * step_samps
        seg = data_arr[:, start:start + win_samps]  # (C, T)

        # Извлечение признаков для каждого канала в текущем окне
        features = extract_channel_features(seg, sfreq=sfreq)  # (C, 7)
        node_features_list.append(torch.tensor(features, dtype=torch.float))

        # Вычисление функциональной связности; здесь используется корреляция
        connectivity = np.corrcoef(seg)
        local_edge_index = build_edge_index(num_channels=n_channels, connectivity_matrix=connectivity,
                                            threshold=threshold)

        # Смещение индексов для текущего окна
        adjusted_edge_index = local_edge_index + cumulative_offset
        edge_indices_list.append(adjusted_edge_index)

        window_indices.append((cumulative_offset, cumulative_offset + n_channels))
        cumulative_offset += n_channels
        window_count += 1

    if len(node_features_list) == 0:
        raise ValueError("Не удалось получить ни одного окна из Raw данных.")

    # Объединение признаков всех окон: итоговый тензор размера (num_windows * n_channels, 7)
    x = torch.cat(node_features_list, dim=0)

    # Объединение локальных ребер всех окон
    if edge_indices_list:
        local_edge_index_all = torch.cat(edge_indices_list, dim=1)
    else:
        local_edge_index_all = torch.empty((2, 0), dtype=torch.long)

    # Добавление временных ребер: связываем узлы, соответствующие одному и тому же каналу из соседних окон
    temporal_edges = []
    for w in range(len(window_indices) - 1):
        start1, _ = window_indices[w]
        start2, _ = window_indices[w + 1]
        for ch in range(n_channels):
            temporal_edges.append([start1 + ch, start2 + ch])
            temporal_edges.append([start2 + ch, start1 + ch])
    if temporal_edges:
        temporal_edge_index = torch.tensor(temporal_edges, dtype=torch.long).t().contiguous()
    else:
        temporal_edge_index = torch.empty((2, 0), dtype=torch.long)

    # Объединение локальных и временных ребер
    edge_index = torch.cat([local_edge_index_all, temporal_edge_index], dim=1)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


# --------------------------------------------------------------------------------
# 3) Основной скрипт
# --------------------------------------------------------------------------------

def main():
    part_tsv = os.path.join(BIDS_ROOT, PARTICIPANTS_TSV)
    if not os.path.isfile(part_tsv):
        raise FileNotFoundError(f"Не найден {part_tsv}")

    participants_df = pd.read_csv(part_tsv, sep='\t')
    all_data = []

    for idx, row in participants_df.iterrows():
        subj_id = row['participant_id']
        group = row['Group']
        if group not in group_map:
            print(f"Пропускаем {subj_id}: группа '{group}' не A/C.")
            continue
        label = group_map[group]
        sub_folder = os.path.join(BIDS_ROOT, subj_id, "eeg")
        set_fname = os.path.join(sub_folder, f"{subj_id}_task-eyesclosed_eeg.set")
        if not os.path.isfile(set_fname):
            print(f"Нет файла: {set_fname}, пропускаем.")
            continue

        try:
            raw = process_one_subject(
                set_fname,
                l_freq=L_FREQ,
                h_freq=H_FREQ,
                do_ica=DO_ICA,
                ica_ncomp=ICA_NCOMP
            )
            # Формируем накопительный граф для данного субъекта,
            # который учитывает как локальные функциональные связи, так и временную динамику.
            cumulative_data = cumulative_graph_from_raw(
                raw,
                win_sec=WINDOW_SEC,
                step_sec=STEP_SEC,
                label=label,
                threshold=0.5
            )
            all_data.append(cumulative_data)
            print(
                f"{subj_id}: сформирован накопительный граф с {cumulative_data.x.shape[0]} узлами. Группа={group} -> label={label}")
        except Exception as ex:
            print(f"Ошибка subj={subj_id}: {ex}")
            continue

    if len(all_data) == 0:
        print("Нет данных для сохранения. all_data пуст.")
        return

    # --------------------------------------------------------------------------------
    # Балансировка классов (для двух классов)
    # --------------------------------------------------------------------------------
    from collections import defaultdict
    class_map = defaultdict(list)
    for d in all_data:
        lbl = int(d.y.item())
        class_map[lbl].append(d)

    print("\nДо балансировки:")
    for lbl, data_list in class_map.items():
        print(f"Class {lbl} = {len(data_list)} графов")

    min_count = min(len(lst) for lst in class_map.values())
    print(f"min_count={min_count} => укорачиваем классы до {min_count} графов.")
    balanced_data = []
    for lbl, data_list in class_map.items():
        random.shuffle(data_list)
        selected = data_list[:min_count]
        balanced_data.extend(selected)
    random.shuffle(balanced_data)

    print("\nПосле балансировки:")
    dist_after = {}
    for d in balanced_data:
        lbl = int(d.y.item())
        dist_after[lbl] = dist_after.get(lbl, 0) + 1
    for lbl in sorted(dist_after.keys()):
        print(f"Class {lbl} = {dist_after[lbl]} графов")

    out_path = os.path.join(BIDS_ROOT, OUT_FILE)
    torch.save(balanced_data, out_path)
    print(f"\nСохранено: {out_path}")
    print(f"Всего графов после балансировки: {len(balanced_data)}")


if __name__ == "__main__":
    DO_ICA = False  # Меняйте по необходимости
    main()
