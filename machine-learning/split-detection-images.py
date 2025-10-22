import os
import shutil
import random
from tqdm import tqdm

# Główne zmienne
DATASET_DIR = "detection-dataset"
IMGS_DIR = os.path.join(DATASET_DIR, "imgs")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

OUTPUT_DIR = "split-detection-dataset"
OUTPUT_IMAGES_TRAIN = os.path.join(OUTPUT_DIR, "images/train")
OUTPUT_IMAGES_VAL = os.path.join(OUTPUT_DIR, "images/val")
OUTPUT_LABELS_TRAIN = os.path.join(OUTPUT_DIR, "labels/train")
OUTPUT_LABELS_VAL = os.path.join(OUTPUT_DIR, "labels/val")

IMAGE_EXTENTIONS = [".jpg", ".jpeg", ".png"]
VAL_SPLIT = 0.2
PROGRESSBAR_FORMAT = (
    "{desc}: |{bar}| {percentage:.1f}% "
    "[Przeanalizowano: {n_fmt}/{total_fmt}, Czas: {elapsed}, Pozostało: {remaining}]"
)

# Funkcja kopiuje obrazy i etykiety do wskazanego zbioru
def copy_files(images_list, target_image_dir, target_label_dir, image_dir_type):
    for image_file in tqdm(images_list, desc=f"Kopiowanie obrazów i etykiet ({image_dir_type})", bar_format=PROGRESSBAR_FORMAT, dynamic_ncols=True):
        shutil.copy(os.path.join(IMGS_DIR, image_file), os.path.join(target_image_dir, image_file))

        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_src = os.path.join(LABELS_DIR, label_file)
        label_dst = os.path.join(target_label_dir, label_file)

        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            print(f"Brak etykiety dla {image_file} - pomijam etykietę.")

# Funkcja przetwarza obrazy i ich etykiety oraz dzieli je na zbiór treningowy i walidacyjny
def main():
    os.makedirs(OUTPUT_IMAGES_TRAIN, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_VAL, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_TRAIN, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_VAL, exist_ok=True)

    images = [f for f in os.listdir(IMGS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    images.sort()

    random.shuffle(images)
    num_val = int(len(images) * VAL_SPLIT)
    val_images = images[:num_val]
    train_images = images[num_val:]

    copy_files(train_images, OUTPUT_IMAGES_TRAIN, OUTPUT_LABELS_TRAIN, "train")
    copy_files(val_images, OUTPUT_IMAGES_VAL, OUTPUT_LABELS_VAL, "val")

    print(f"GOTOWE! Zbiór treningowy: {len(train_images)} obrazów, Zbiór walidacyjny: {len(val_images)} obrazów")

if __name__ == "__main__":
    main()