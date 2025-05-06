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
OUT_FILE = "my_eeg_dataset.pt"  # <-- Итоговый файл с List[Data]
PARTICIPANTS_TSV = "participants.tsv"

# Частоты фильтра
L_FREQ, H_FREQ = 0.5, 45.0

# Настройки ICA
ICA_NCOMP = 19  # Число компонентов
DO_ICA = True  # Применяем ICA

# Окна
WINDOW_SEC = 5.0
STEP_SEC = 2.0

# Порог для формирования динамического графа по функциональной связности
CONNECTIVITY_THRESHOLD = 0.5

# Маппинг групп (A, C, F) -> метки
# A: Alzheimer, F: MCI/прочие, C: Control
group_map = {
    'A': 1,
    'F': 2,
    'C': 0
}


# --------------------------------------------------------------------------------
# 2) Функции для построения графа и обработки данных
# --------------------------------------------------------------------------------

def build_dynamic_edge_index(connectivity_matrix, threshold=0.5):
    """
    Строит edge_index на основе матрицы функциональной связности.
    Для каждой пары каналов, если корреляция выше threshold, создается ребро.
    Если ребер не найдено, добавляются self-loop ребра.
    """
    num_channels = connectivity_matrix.shape[0]
    edge_indices = []
    for i in range(num_channels):
        for j in range(num_channels):
            if i != j and connectivity_matrix[i, j] > threshold:
                edge_indices.append([i, j])
    if not edge_indices:
        # Если нет ребер, добавляем self-loops
        edge_indices = [[i, i] for i in range(num_channels)]
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    return edge_index


def process_one_subject(set_fname, l_freq=1., h_freq=40., do_ica=False, ica_ncomp=15):
    """
    Открывает .set-файл, фильтрует, (опц.) ICA, возвращает объект Raw (MNE).
    """
    print(f"Loading: {set_fname}")
    raw = mne.io.read_raw_eeglab(set_fname, preload=True)

    # Фильтрация
    print(f"Filtering {l_freq}–{h_freq} Hz")
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    # Реферирование (average reference)
    raw.set_eeg_reference('average', projection=False)

    # (Опционально) ICA
    if do_ica:
        print(f"Applying ICA with {ica_ncomp} components.")
        ica = mne.preprocessing.ICA(n_components=ica_ncomp, random_state=42)
        ica.fit(raw)
        # В реальности следует анализировать ica.exclude
        raw = ica.apply(raw)

    return raw


def extract_channel_features(seg, sfreq):
    """
    seg: (C, T) — C=число каналов, T=число отсчетов.
    Возвращает матрицу (C, F) с набором признаков:
      - mean, std,
      - delta (1-4 Hz), theta (4-8), alpha (8-12), beta (12-30), gamma (30-45)
    Итоговое число признаков F=7.
    Для удобства обучения применяется масштабирование (x1e5).
    """
    import mne
    n_channels, n_times = seg.shape

    # 1) mean, std
    seg_mean = seg.mean(axis=1)  # (C,)
    seg_std = seg.std(axis=1)  # (C,)

    # 2) PSD через Welch
    psds, freqs = mne.time_frequency.psd_array_welch(
        seg, sfreq=sfreq, fmin=0.5, fmax=45, n_fft=256
    )

    # Маски для разных диапазонов
    delta_mask = (freqs >= 1) & (freqs < 4)
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs < 12)
    beta_mask = (freqs >= 12) & (freqs < 30)
    gamma_mask = (freqs >= 30) & (freqs <= 45)

    delta_power = psds[:, delta_mask].mean(axis=1)  # (C,)
    theta_power = psds[:, theta_mask].mean(axis=1)
    alpha_power = psds[:, alpha_mask].mean(axis=1)
    beta_power = psds[:, beta_mask].mean(axis=1)
    gamma_power = psds[:, gamma_mask].mean(axis=1)

    # Собираем 7 признаков
    features = np.stack(
        [seg_mean, seg_std, delta_power, theta_power, alpha_power, beta_power, gamma_power],
        axis=1
    )  # shape=(C, 7)

    # 3) Масштабирование для удобства обучения
    features = features * 1e5

    return features  # (C, 7)


def segment_to_data(raw, win_sec=5.0, step_sec=2.0, label=0):
    """
    Разбивает Raw на окна (псевдо real-time).
    Для каждого окна:
      - Вычисляется матрица функциональной связности (np.corrcoef) по каналам.
      - Строится динамический edge_index на основе connectivity.
      - Вычисляются признаки по каналам через extract_channel_features.
    Возвращается список объектов Data с динамической топологией.
    """
    sfreq = raw.info['sfreq']
    data = raw.get_data()  # (C, N)
    n_channels = data.shape[0]

    win_samps = int(win_sec * sfreq)
    step_samps = int(step_sec * sfreq)
    n_times = data.shape[1]

    data_list = []
    start = 0
    while (start + win_samps) <= n_times:
        seg = data[:, start:start + win_samps]  # (C, T)

        # Вычисляем матрицу функциональной связности (корреляции между каналами)
        connectivity = np.corrcoef(seg)
        dynamic_edge_index = build_dynamic_edge_index(connectivity, threshold=CONNECTIVITY_THRESHOLD)

        # Вычисляем признаки по каналам
        features = extract_channel_features(seg, sfreq=sfreq)
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)

        d = Data(x=x, edge_index=dynamic_edge_index, y=y)
        data_list.append(d)

        start += step_samps

    return data_list


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
            print(f"Пропускаем {subj_id}: неизвестная группа '{group}'.")
            continue
        label = group_map[group]

        sub_folder = os.path.join(BIDS_ROOT, subj_id, "eeg")
        set_fname = os.path.join(sub_folder, f"{subj_id}_task-eyesclosed_eeg.set")

        if not os.path.isfile(set_fname):
            print(f"Нет файла: {set_fname}, пропускаем.")
            continue

        try:
            # 1) Обрабатываем (фильтр + опц. ICA)
            raw = process_one_subject(
                set_fname,
                l_freq=L_FREQ,
                h_freq=H_FREQ,
                do_ica=DO_ICA,
                ica_ncomp=ICA_NCOMP
            )

            # 2) Разбиваем на окна с динамическим построением графа
            data_list = segment_to_data(
                raw,
                win_sec=WINDOW_SEC,
                step_sec=STEP_SEC,
                label=label
            )

            all_data.extend(data_list)
            print(f"{subj_id}: {len(data_list)} окон. Группа={group} -> label={label}")
        except Exception as ex:
            print(f"Ошибка subj={subj_id}: {ex}")
            continue

    if len(all_data) == 0:
        print("Нет данных для сохранения. all_data пуст.")
        return

    # --------------------------------------------------------------------------------
    # БАЛАНСИРОВКА КЛАССОВ
    # --------------------------------------------------------------------------------
    from collections import defaultdict
    class_map = defaultdict(list)
    for d in all_data:
        lbl = int(d.y.item())
        class_map[lbl].append(d)

    print("До балансировки:")
    for lbl, data_list in class_map.items():
        print(f"Class {lbl} = {len(data_list)} окон")

    # Downsample к min_count
    min_count = min(len(lst) for lst in class_map.values())
    print(f"min_count={min_count} (будем укорачивать классы до {min_count} окон)")

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
        print(f"Class {lbl} = {dist_after[lbl]} окон")

    out_path = os.path.join(BIDS_ROOT, OUT_FILE)
    torch.save(balanced_data, out_path)
    print(f"\nСохранено: {out_path}")
    print(f"Всего сегментов после балансировки: {len(balanced_data)}")


if __name__ == "__main__":
    DO_ICA = False  # Меняйте при необходимости
    main()
