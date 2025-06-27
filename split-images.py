import os
import shutil
import random
from tqdm import tqdm

def get_full_path(*path_args):
    return os.path.join(os.path.dirname(__file__), *path_args)

source_dir = "processed-classification-dataset"
target_dir = "split-dataset"
splits = { "train": 500, "val": 100, "test": 100 }
bar_format = (
    "{desc}: |{bar}| {percentage:.1f}% "
    "[{n_fmt}/{total_fmt} obrazy przeanalizowane, "
    "Czas: {elapsed}, "
    "Pozostało: {remaining}, "
    "Prędkość: {rate_fmt}]"
)

# Tworzenie folderów docelowych
for split in splits:
    for class_name in os.listdir(get_full_path(source_dir)):
        os.makedirs(get_full_path(target_dir, split, class_name), exist_ok=True)

# Przetwarzanie klas
for class_name in tqdm(os.listdir(get_full_path(source_dir)), desc="Przetwarzanie klas", bar_format=bar_format, dynamic_ncols=True):
    class_path = get_full_path(source_dir, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    start = 0
    for split, count in splits.items():
        split_images = images[start:start + count]
        for image in tqdm(split_images, desc=f"Kopiowanie klasy {class_name} ({split})", bar_format=bar_format, dynamic_ncols=True, leave=False):
            src = os.path.join(class_path, image)
            dst = get_full_path(target_dir, split, class_name, image)
            shutil.copy(src, dst)
        start += count

print("GOTOWE! Obrazy zostały podzielone na zbiór treningowy, walidacyjny i testowy")