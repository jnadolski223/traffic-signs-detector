import os
import shutil
import random
from tqdm import tqdm

SOURCE_DIR_NAME = "processed-classification-dataset"
TARGET_DIR_NAME = "split-classification-dataset"
SPLITS = { "train": 512, "val": 128, "test": 128 }
PROGRESSBAR_FORMAT = (
    "{desc}: |{bar}| {percentage:.1f}% "
    "[Przeanalizowano: {n_fmt}/{total_fmt}, Czas: {elapsed}, Pozostało: {remaining}]"
)

# Funkcja zwracająca ścieżkę bezwzględną do pliku/folderu
def get_full_path(*path_args):
    return os.path.join(os.path.dirname(__file__), *path_args)

# Funkcja tworzy foldery docelowe dla obrazów z podziałem na zbiory
def create_target_dirs():
    for split in SPLITS:
        for class_name in os.listdir(get_full_path(SOURCE_DIR_NAME)):
            class_path = get_full_path(SOURCE_DIR_NAME, class_name)
            if os.path.isdir(class_path):
                os.makedirs(get_full_path(TARGET_DIR_NAME, split, class_name), exist_ok=True)

# Funkcja kopiuje obrazy danej klasy do wskazanego zbioru
def copy_images_to_split(class_name, split, images):
    src_class_path = get_full_path(SOURCE_DIR_NAME, class_name)
    dst_class_path = get_full_path(TARGET_DIR_NAME, split, class_name)
    
    for image_name in tqdm(images, desc=f"Kopiowanie klasy {class_name} ({split})", bar_format=PROGRESSBAR_FORMAT, dynamic_ncols=True, leave=False):
        src = os.path.join(src_class_path, image_name)
        dst = os.path.join(dst_class_path, image_name)
        shutil.copy(src, dst)

# Funkcja rozdziela obrazy danej klasy na zbiory
def split_images_for_class(class_name):
    class_path = get_full_path(SOURCE_DIR_NAME, class_name)
    class_images = os.listdir(class_path)
    random.shuffle(class_images)

    start = 0
    for split, count in SPLITS.items():
        split_images = class_images[start:start + count]
        copy_images_to_split(class_name, split, split_images)
        start += count

# Funkcja przetwarza obrazy z całego datasetu i rozdziela na osobne zbiory
def main():
    create_target_dirs()
    class_list = [
        class_dir for class_dir in os.listdir(get_full_path(SOURCE_DIR_NAME)) 
        if os.path.isdir(get_full_path(SOURCE_DIR_NAME, class_dir))
    ]

    for class_name in tqdm(class_list, desc="Przetwarzanie klas", bar_format=PROGRESSBAR_FORMAT, dynamic_ncols=True):
        split_images_for_class(class_name)

    print("GOTOWE! Zakończono przetwarzanie obrazów. Obrazy zostały podzielone na zbiór treningowy, walidacyjny i testowy")

if __name__ == "__main__":
    main()