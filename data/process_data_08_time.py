#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess EEG (OpenNeuro ds004504) ─ фильтрация, ресэмплинг, (опц.) ICA,
нормализация клипов и сохранение их в .npz.
Добавлено: установка стандартного 10-20 монтажа со встроенными xyz-координатами.
"""

import os
import numpy as np
import pandas as pd
import argparse
import mne

# ─────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────
def normalize_clip(clip: np.ndarray) -> np.ndarray:
    """
    Z-score по каждому каналу.
    clip: ndarray [n_channels, n_samples]
    """
    means = clip.mean(axis=1, keepdims=True)
    stds  = clip.std(axis=1, keepdims=True) + 1e-6
    return (clip - means) / stds


def process_one_subject(set_fname: str,
                        l_freq: float = 0.5,
                        h_freq: float = 45.0,
                        resample_rate: int = 200,
                        do_ica: bool = False,
                        ica_ncomp: int = 19) -> mne.io.BaseRaw:
    """
    Читает .set, ставит стандартный монтаж 10-20, фильтрует,
    (опц.) применяет ICA и ресэмплирует.
    Возвращает объект Raw.
    """
    print(f"\n🠺 Loading: {set_fname}")
    raw = mne.io.read_raw_eeglab(set_fname, preload=True)

    # ── ДОБАВЛЕН МОНТАЖ 10-20 СО ВСЕМИ xyz-КООРДИНАТАМИ ──────────────────
    montage = mne.channels.make_standard_montage('standard_1020')
    # Сверяем названия каналов (в датасете они уже как Fp1, Fp2 …)
    raw.set_montage(montage, on_missing='ignore')
    print("✓ Standard 10-20 montage applied with xyz coordinates.")

    # ───────────────────────────────────────────────────────────────────────
    print(f"✓ Filtering {l_freq}–{h_freq} Hz")
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    raw.set_eeg_reference('average', projection=False)

    if do_ica:
        print(f"✓ ICA ({ica_ncomp} components)")
        ica = mne.preprocessing.ICA(n_components=ica_ncomp,
                                    random_state=42,
                                    max_iter='auto')
        ica.fit(raw)
        raw = ica.apply(raw)

    print(f"✓ Resampling → {resample_rate} Hz")
    raw.resample(resample_rate)
    return raw


def segment_subject(raw: mne.io.BaseRaw,
                    clip_len: float = 60.0,
                    step_size: float = 60.0,
                    normalize: bool = True):
    """
    Делит Raw на клипы длиной clip_len секунд.
    Возвращает список клипов + sfreq.
    """
    sfreq = raw.info['sfreq']
    data  = raw.get_data()                     # [n_channels, n_times]
    clip_samples = int(clip_len  * sfreq)
    step_samples = int(step_size * sfreq)

    clips = []
    for start in range(0, data.shape[1] - clip_samples + 1, step_samples):
        clip = data[:, start:start + clip_samples]
        if normalize:
            clip = normalize_clip(clip)
        clips.append(clip)

    return clips, sfreq


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ds004504 EEG data with 10-20 montage")
    parser.add_argument("--bids_root", type=str,
                        default="ds004504",
                        help="Корень BIDS-датасета")
    parser.add_argument("--participants_tsv", type=str,
                        default="participants.tsv",
                        help="Имя participants.tsv (относительно bids_root)")
    parser.add_argument("--output_dir", type=str,
                        default="preproc_clips",
                        help="Куда сохранять .npz-клипы")
    parser.add_argument("--clip_len", type=float, default=60.0,
                        help="Длина клипа, сек")
    parser.add_argument("--step_size", type=float, default=60.0,
                        help="Шаг окна, сек")
    parser.add_argument("--resample_rate", type=int, default=200,
                        help="Частота ресэмплинга, Гц")
    parser.add_argument("--do_ica", action="store_true",
                        help="Включить ICA артефакт-ремувал")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Загружаем participants.tsv ─────────────────────────────────────────
    participants_path = os.path.join(args.bids_root, args.participants_tsv)
    if not os.path.isfile(participants_path):
        raise FileNotFoundError(f"❌ Нет файла participants.tsv: {participants_path}")

    participants_df = pd.read_csv(participants_path, sep='\t')

    # ── Обрабатываем каждого участника ────────────────────────────────────
    for _, row in participants_df.iterrows():
        subj_id = row['participant_id']
        group   = row['Group']            # 'A', 'C', 'F' …
        if group not in ('A', 'C'):
            print(f"⏩ {subj_id}: группа '{group}' не A/C, пропускаем.")
            continue

        label = 1 if group == 'A' else 0
        subj_folder = os.path.join(args.bids_root, subj_id, "eeg")
        set_fname   = os.path.join(subj_folder,
                                   f"{subj_id}_task-eyesclosed_eeg.set")
        if not os.path.isfile(set_fname):
            print(f"⚠️  Нет файла {set_fname} — пропуск.")
            continue

        try:
            raw = process_one_subject(set_fname,
                                      l_freq=0.5,
                                      h_freq=45.0,
                                      resample_rate=args.resample_rate,
                                      do_ica=args.do_ica,
                                      ica_ncomp=19)

            clips, sfreq = segment_subject(raw,
                                           clip_len=args.clip_len,
                                           step_size=args.step_size,
                                           normalize=True)

            subj_out_dir = os.path.join(args.output_dir, subj_id)
            os.makedirs(subj_out_dir, exist_ok=True)

            print(f"✓ {subj_id}: {len(clips)} клипов. Label={label}")

            for i, clip in enumerate(clips):
                np.savez(os.path.join(subj_out_dir,
                                      f"{subj_id}_clip_{i}.npz"),
                         clip=clip,
                         sfreq=sfreq,
                         label=label)

        except Exception as exc:
            print(f"❌ Ошибка {subj_id}: {exc}")
            continue


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
