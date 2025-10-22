import os
import shutil
import random
import imageio
import cv2
import albumentations as A
from tqdm import tqdm

# Główne zmienne
DATASET_NAME_OLD = "classification-dataset"
DATASET_NAME_NEW = "processed-classification-dataset"
IMAGE_EXTENTIONS = [".jpg", ".jpeg", ".png"]
CLASS_IMAGE_LIMIT = 768  # trening - 512, walidacja - 128, test - 128
TARGET_IMAGE_SIZE = (224, 224)
PROGRESSBAR_FORMAT = (
    "{desc}: |{bar}| {percentage:.1f}% "
    "[Przeanalizowano: {n_fmt}/{total_fmt}, Czas: {elapsed}, Pozostało: {remaining}]"
)

# Konfiguracja modelu do augmentacji obrazów
augmenter = A.Compose([
    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.5),
    A.MotionBlur(p=0.2, blur_limit=3),
    A.CLAHE(p=0.3, clip_limit=(1, 4)),
    A.HueSaturationValue(p=0.3, hue_shift_limit=3, sat_shift_limit=10, val_shift_limit=10),
    A.MultiplicativeNoise(p=0.5, multiplier=(0.9, 1.1)),
])

# Funkcja kopiuje obrazy oraz zmienia ich nazwy na format {class_name}-{i}{image_extension}
def copy_and_rename_images(class_name, images):
    class_path_old = os.path.join(DATASET_NAME_OLD, class_name)
    class_path_new = os.path.join(DATASET_NAME_NEW, class_name)
    os.makedirs(class_path_new, exist_ok=True)
    
    copied_count = 0
    for i, image_name_old in tqdm(enumerate(images), total=len(images), desc=f"Kopiowanie klasy {class_name}", bar_format=PROGRESSBAR_FORMAT, dynamic_ncols=True):
        image_extention = os.path.splitext(image_name_old)[1].lower()
        image_name_new = f"{class_name}-{i+1}{image_extention}"
        shutil.copy(os.path.join(class_path_old, image_name_old), os.path.join(class_path_new, image_name_new))
        copied_count += 1
    
    return copied_count

# Funckja augmentuje obrazy do podanego limitu obrazów na klasę
def augment_images(class_name, existing_count):
    missing = CLASS_IMAGE_LIMIT - existing_count
    if missing <= 0:
        return 0
    
    class_path_new = os.path.join(DATASET_NAME_NEW, class_name)
    original_images = [os.path.join(class_path_new, image_name) for image_name in os.listdir(class_path_new)]

    augmented_count = 0
    for i in tqdm(range(missing), desc=f"Augmentacja klasy {class_name}", bar_format=PROGRESSBAR_FORMAT, dynamic_ncols=True):
        selected_image_path = original_images[i % existing_count]
        image_content = imageio.v2.imread(selected_image_path)
        augmented_image = augmenter(image=image_content)["image"]
        augmented_image_name = f"{class_name}-{existing_count + i + 1}.jpg"
        imageio.imwrite(os.path.join(class_path_new, augmented_image_name), augmented_image)
        augmented_count += 1
    
    return augmented_count

# Funkcja skaluje obrazy do wybranego rozmiaru
def resize_images(class_name):
    class_path = os.path.join(DATASET_NAME_NEW, class_name)

    resized_count = 0
    for image_name in tqdm(os.listdir(class_path), desc=f"Skalowanie klasy {class_name}", bar_format=PROGRESSBAR_FORMAT, dynamic_ncols=True):
        image_path = os.path.join(class_path, image_name)
        image_content = imageio.v2.imread(image_path)
        resized_image = cv2.resize(image_content, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        imageio.imwrite(image_path, resized_image)
        resized_count += 1
    
    return resized_count

# Funkcja przetwarza obrazy: kopiowanie, zmiana nazwy, augmentacja, skalowanie
def process_class(class_name, idx, total):
    class_images = [
        image_name for image_name in os.listdir(os.path.join(DATASET_NAME_OLD, class_name)) 
        if os.path.splitext(image_name)[1].lower() in IMAGE_EXTENTIONS
    ]
    
    selected_class_images = random.sample(class_images, min(CLASS_IMAGE_LIMIT, len(class_images)))
    copied_count = copy_and_rename_images(class_name, selected_class_images)
    augmented_count = augment_images(class_name, len(selected_class_images))
    resized_count = resize_images(class_name)
    print(f"[{idx}/{total}] Klasa: {class_name}, Skopiowane obrazy: {copied_count}, Zaugmentowane obrazy: {augmented_count}, Przeskalowane obrazy: {resized_count}\n")

# Funkcja przetwarza obrazy z całego datasetu i tworzy przetworzoną kopię
def main():
    os.makedirs(DATASET_NAME_NEW, exist_ok=True)
    class_list = [
        class_dir for class_dir in os.listdir(DATASET_NAME_OLD) 
        if os.path.isdir(os.path.join(DATASET_NAME_OLD, class_dir))
    ]

    for idx, class_name in enumerate(class_list, 1):
        process_class(class_name, idx, len(class_list))
    
    print(f"GOTOWE! Zakończono przetwarzanie obrazów. Każda klasa zawiera po {CLASS_IMAGE_LIMIT} obrazów o rozmirach {TARGET_IMAGE_SIZE[0]}x{TARGET_IMAGE_SIZE[1]}")

if __name__ == "__main__":
    main()