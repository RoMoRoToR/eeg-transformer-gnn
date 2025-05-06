import os
import numpy as np
import pandas as pd
import torch
import mne
import argparse


def normalize_clip(clip):
    """
    Применяет z-score нормализацию по каждому каналу.
    clip: numpy array [n_channels, n_samples]
    """
    means = clip.mean(axis=1, keepdims=True)
    stds = clip.std(axis=1, keepdims=True) + 1e-6
    return (clip - means) / stds


def process_one_subject(set_fname, l_freq=0.5, h_freq=45.0, resample_rate=200,
                        do_filter=False, do_ica=False, ica_ncomp=19):
    """
    Загружает .set файл EEG.
    Если do_filter=True, применяется фильтрация (l_freq-h_freq), референсирование,
    опционально ICA, а затем сигнал ресемплируется до resample_rate.
    Если do_filter=False, данные не фильтруются (но выполняется референсирование и ресемплирование).
    Возвращает объект Raw (MNE).
    """
    print(f"Loading: {set_fname}")
    raw = mne.io.read_raw_eeglab(set_fname, preload=True)

    if do_filter:
        print(f"Filtering {l_freq}–{h_freq} Hz")
        raw.filter(l_freq=l_freq, h_freq=h_freq)
    else:
        print("Skipping filtering: using raw data without bandpass filtering.")

    raw.set_eeg_reference('average', projection=False)

    if do_ica:
        print(f"Applying ICA with {ica_ncomp} components.")
        ica = mne.preprocessing.ICA(n_components=ica_ncomp, random_state=42)
        ica.fit(raw)
        raw = ica.apply(raw)

    print(f"Resampling to {resample_rate} Hz")
    raw.resample(resample_rate)
    return raw


def segment_subject(raw, clip_len=60.0, step_size=60.0, normalize=True):
    """
    Разбивает сигнал Raw на клипы заданной длительности (в секундах) с указанным шагом.
    Возвращает список клипов (каждый клип – массив [n_channels, n_samples]).
    Если normalize=True, применяется z-score нормализация по каждому каналу.
    """
    sfreq = raw.info['sfreq']
    data = raw.get_data()  # shape: (n_channels, n_times)
    clip_samples = int(clip_len * sfreq)
    step_samples = int(step_size * sfreq)
    clips = []
    for start in range(0, data.shape[1] - clip_samples + 1, step_samples):
        clip = data[:, start:start + clip_samples]
        if normalize:
            clip = normalize_clip(clip)
        clips.append(clip)
    return clips, sfreq


def main():
    parser = argparse.ArgumentParser(description="Preprocess EEG data for offline training")
    parser.add_argument("--bids_root", type=str,
                        default="/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/ds004504",
                        help="BIDS root directory")
    parser.add_argument("--participants_tsv", type=str, default="participants.tsv",
                        help="Participants TSV file")
    parser.add_argument("--output_dir", type=str, default="preproc_clips_no_filter",
                        help="Directory to save preprocessed clips")
    parser.add_argument("--clip_len", type=float, default=60.0, help="Clip length in seconds")
    parser.add_argument("--step_size", type=float, default=60.0, help="Step size for segmentation")
    parser.add_argument("--resample_rate", type=int, default=200, help="Resample rate (Hz)")
    # Флаг для фильтрации: если False, данные не фильтруются (исходные)
    parser.add_argument("--do_filter", action="store_true", help="Apply bandpass filtering")
    parser.add_argument("--do_ica", action="store_true", help="Apply ICA for artifact removal")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    participants_path = os.path.join(args.bids_root, args.participants_tsv)
    if not os.path.isfile(participants_path):
        raise FileNotFoundError(f"Participants TSV file not found: {participants_path}")
    participants_df = pd.read_csv(participants_path, sep='\t')

    for idx, row in participants_df.iterrows():
        subj_id = row['participant_id']
        group = row['Group']
        # Оставляем только группы Alzheimer (A) и Control (C)
        if group not in ['A', 'C']:
            print(f"Пропускаем {subj_id}: группа '{group}' не A/C.")
            continue
        label = 1 if group == 'A' else 0

        subj_folder = os.path.join(args.bids_root, subj_id, "eeg")
        set_fname = os.path.join(subj_folder, f"{subj_id}_task-eyesclosed_eeg.set")
        if not os.path.isfile(set_fname):
            print(f"Нет файла: {set_fname}. Пропускаем {subj_id}.")
            continue

        try:
            # Если хотим тестировать на исходных (нефильтрованных) данных, запускаем с do_filter=False
            raw = process_one_subject(set_fname, l_freq=0.5, h_freq=45.0,
                                      resample_rate=args.resample_rate,
                                      do_filter=False, do_ica=args.do_ica, ica_ncomp=19)
            clips, sfreq = segment_subject(raw, clip_len=args.clip_len,
                                           step_size=args.step_size, normalize=True)
            subj_out_dir = os.path.join(args.output_dir, subj_id)
            os.makedirs(subj_out_dir, exist_ok=True)
            print(f"{subj_id}: найдено {len(clips)} клипов. Группа: {group}, метка: {label}")
            for i, clip in enumerate(clips):
                out_file = os.path.join(subj_out_dir, f"{subj_id}_clip_{i}.npz")
                np.savez(out_file, clip=clip, sfreq=sfreq, label=label)
            print(f"Сохранено {len(clips)} клипов для {subj_id}.")
        except Exception as e:
            print(f"Ошибка при обработке {subj_id}: {e}")
            continue


if __name__ == "__main__":
    main()
