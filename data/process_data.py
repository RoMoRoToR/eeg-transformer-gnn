import os
import numpy as np
import pandas as pd
import torch
import mne

# --------------------------------------------------------------------------------
# 1) Настройки
# --------------------------------------------------------------------------------

BIDS_ROOT = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/ds004504"  # <-- Укажите путь к корню датасета (где sub-XXX и participants.tsv)
OUT_FILE = "my_eeg_dataset.pt"  # <-- Итоговый файл с (X, y)
PARTICIPANTS_TSV = "participants.tsv"

# Частоты фильтра
L_FREQ, H_FREQ = 0.5, 45.0

# Настройки ICA
ICA_NCOMP = 19  # Число компонентов
DO_ICA = True  # Применяем ICA

# Окна
WINDOW_SEC = 5.0
STEP_SEC = 2.0

# Уровни квантования
AMP_QUANT_LEVELS = 512

# Маппинг групп (A, C, F) -> метки
# A: Alzheimer, C: Control, F: "прочие" (например, Mild Cognitive Impairment и т.п.)
group_map = {
    'A': 1,
    'F': 2,
    'C': 0
}


# --------------------------------------------------------------------------------
# 2) Функции
# --------------------------------------------------------------------------------

def amplitude_quantization(seg, levels=256):
    """
    Упрощённая скалярная квантизация амплитуд.
    seg: (C, T) numpy array
    1) min-max по сегменту
    2) нормировка => [0..1]
    3) масштабирование => [0..levels)
    4) округление
    5) clip в [0..levels-1]
    Возвращает 1D numpy array int (длина = C*T).
    """
    seg_flat = seg.flatten()
    val_min, val_max = seg_flat.min(), seg_flat.max()
    if abs(val_min - val_max) < 1e-12:
        # Сигнал почти константный
        return np.zeros_like(seg_flat, dtype=int)
    # нормируем в [0..1]
    seg_norm = (seg_flat - val_min) / (val_max - val_min)
    # масштабируем в [0..levels)
    seg_q = np.floor(seg_norm * levels).astype(int)
    seg_q = np.clip(seg_q, 0, levels - 1)
    return seg_q


def process_one_subject(set_fname, l_freq=1., h_freq=40., do_ica=False, ica_ncomp=15):
    """
    Открывает .set-файл, фильтрует, (опционально) ICA, возвращает Raw (MNE).
    """
    # Загружаем
    print(f"Loading: {set_fname}")
    raw = mne.io.read_raw_eeglab(set_fname, preload=True)

    # Фильтрация
    print(f"Filtering {l_freq}–{h_freq} Hz")
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    # Реферирование по среднему
    raw.set_eeg_reference('average', projection=False)

    # (Опционально) ICA
    if do_ica:
        print(f"Applying ICA with {ica_ncomp} components (but not excluding any comps).")
        ica = mne.preprocessing.ICA(n_components=ica_ncomp, random_state=42)
        ica.fit(raw)
        # В реальности нужно смотреть компоненты и exclude шумные
        raw = ica.apply(raw)

    return raw


def segment_and_quantize(raw, win_sec=2.0, step_sec=1.0, levels=256):
    """
    Нарезка на окна + квантование:
    Возвращает список numpy.array(shape=(C*T,)) -- «последовательности токенов».
    """
    sfreq = raw.info['sfreq']
    data = raw.get_data()  # (C, N)
    n_times = data.shape[1]

    win_samps = int(win_sec * sfreq)
    step_samps = int(step_sec * sfreq)

    segments = []
    start = 0
    while (start + win_samps) <= n_times:
        seg = data[:, start:start + win_samps]  # (C, T)
        # квантование
        seg_q = amplitude_quantization(seg, levels=levels)
        segments.append(seg_q)
        start += step_samps
    return segments


# --------------------------------------------------------------------------------
# 3) Основной скрипт
# --------------------------------------------------------------------------------

def main():
    # Загрузим participants.tsv
    part_tsv = os.path.join(BIDS_ROOT, PARTICIPANTS_TSV)
    if not os.path.isfile(part_tsv):
        raise FileNotFoundError(f"Не найден {part_tsv}")
    participants_df = pd.read_csv(part_tsv, sep='\t')

    # Общие списки
    all_X = []
    all_y = []

    # Идём по всем строчкам participants.tsv
    for idx, row in participants_df.iterrows():
        subj_id = row['participant_id']  # например, 'sub-001'
        group = row['Group']  # 'A', 'C', 'F', ...
        # Проверим, есть ли в словаре group_map
        if group not in group_map:
            print(f"Пропускаем {subj_id}: неизвестная группа '{group}'.")
            continue
        label = group_map[group]

        # Путь к .set-файлу
        # Допустим, task="eyesclosed". Убедитесь, что это правильно для всех
        # или, если есть разные tasks, используйте row['...'] или ищите файл с os.listdir.
        sub_folder = os.path.join(BIDS_ROOT, subj_id, "eeg")

        # По вашему скриншоту, формат: sub-001_task-eyesclosed_eeg.set
        # Но могли быть иные (eyesopen или rest). Тут мы предположим eyesclosed.
        set_fname = os.path.join(sub_folder, f"{subj_id}_task-eyesclosed_eeg.set")

        if not os.path.isfile(set_fname):
            print(f"Нет файла: {set_fname}, пропускаем.")
            continue

        try:
            # 1) обрабатываем (фильтр + опц. ICA)
            raw = process_one_subject(
                set_fname,
                l_freq=L_FREQ,
                h_freq=H_FREQ,
                do_ica=DO_ICA,
                ica_ncomp=ICA_NCOMP
            )
            # 2) сегментация + квантование
            seg_list = segment_and_quantize(raw,
                                            win_sec=WINDOW_SEC,
                                            step_sec=STEP_SEC,
                                            levels=AMP_QUANT_LEVELS)
            # seg_list: list of np.array, each shape=(C*T,)

            # 3) добавляем в общий X, y
            for seg_q in seg_list:
                all_X.append(seg_q)
                all_y.append(label)

            print(f"{subj_id}: {len(seg_list)} окон. Группа={group} -> label={label}")
        except Exception as ex:
            print(f"Ошибка subj={subj_id}: {ex}")
            continue

    # --------------------------------------------------------------------------------
    # Превращаем в PyTorch-тензоры
    # --------------------------------------------------------------------------------
    if len(all_X) == 0:
        print("Нет данных для сохранения. all_X пуст.")
        return

    # all_X - список numpy (C*T,) одинаковой длины?
    # Предположим, что во всех испытуемых одинаковое кол-во каналов
    # и частота дискретизации, значит C*T одинаково.
    X_array = np.stack(all_X, axis=0)  # shape (N, C*T)
    X_tensor = torch.from_numpy(X_array).long()  # (N, seq_len)
    y_tensor = torch.tensor(all_y, dtype=torch.long)  # (N,)

    print(f"\nИтого всего окон: {X_tensor.shape[0]}")
    print(f"Размер одного сегмента: {X_tensor.shape[1]} (C*T)")
    print("Пример меток (y):", set(y_tensor.tolist()))

    # Сохраняем
    out_path = os.path.join(BIDS_ROOT, OUT_FILE)
    torch.save((X_tensor, y_tensor), out_path)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    # Если нужен ICA - меняйте:
    DO_ICA = False
    main()
