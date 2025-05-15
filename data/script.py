import os
import shutil
import random

def split_data_by_subject(root_dir, train_dir, test_dir, split_ratio=0.8, seed=42):
    """
    Делит данные по subject-wise на train и test и копирует папки с клипами.

    :param root_dir: исходная папка с папками субъектов, содержащими .npz
    :param train_dir: папка для тренировочных субъектов
    :param test_dir: папка для тестовых субъектов
    :param split_ratio: доля субъектов для train (по умолчанию 0.8)
    :param seed: сид для воспроизводимости
    """
    # Получаем список субъектов
    subjects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    random.seed(seed)
    random.shuffle(subjects)

    # Разбиваем
    n_train = int(len(subjects) * split_ratio)
    train_subjs = subjects[:n_train]
    test_subjs  = subjects[n_train:]

    # Создаём выходные директории
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Копируем train
    for subj in train_subjs:
        src = os.path.join(root_dir, subj)
        dst = os.path.join(train_dir, subj)
        shutil.copytree(src, dst)
    # Копируем test
    for subj in test_subjs:
        src = os.path.join(root_dir, subj)
        dst = os.path.join(test_dir, subj)
        shutil.copytree(src, dst)

    print(f"Done! Train subjects: {len(train_subjs)}, Test subjects: {len(test_subjs)}")

if __name__ == "__main__":
    ROOT_DIR = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips"
    TRAIN_DIR = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips_train"
    TEST_DIR  = "/Users/taniyashuba/PycharmProjects/eeg-transformer-gnn/data/preproc_clips_test"
    split_data_by_subject(ROOT_DIR, TRAIN_DIR, TEST_DIR)
