import os
import shutil
import random
import imageio
import cv2
import albumentations as A
from tqdm import tqdm

def get_full_path(*path_args):
    return os.path.join(os.path.dirname(__file__), *path_args)

# Konfiguracja modelu do augmentacji obrazów
augmenter = A.Compose([
    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.5),
    A.MotionBlur(p=0.2, blur_limit=3),
    A.CLAHE(p=0.3, clip_limit=(1, 4)),
    A.HueSaturationValue(p=0.3, hue_shift_limit=3, sat_shift_limit=10, val_shift_limit=10),
    A.MultiplicativeNoise(p=0.5, multiplier=(0.9, 1.1)),
])

# Główne zmienne
dataset_dir_current = "classification-dataset"
dataset_dir_new = "processed-classification-dataset"
img_extentions = [".jpg", ".jpeg", ".png"]
img_class_limit = 700
target_size = (64, 64)
bar_format = (
    "{desc}: |{bar}| {percentage:.1f}% "
    "[{n_fmt}/{total_fmt} obrazy przeanalizowane, "
    "Czas: {elapsed}, "
    "Pozostało: {remaining}, "
    "Prędkość: {rate_fmt}]"
)

# Sprawdzanie folderu z datasetem
print(f"Sprawdzam folder {get_full_path(dataset_dir_current)}")
class_dirs = os.listdir(get_full_path(dataset_dir_current))
print(f"Liczba klas obrazów: {len(class_dirs)}")

# Tworzenie folderu wyjściowego dla nowego datasetu
os.makedirs(get_full_path(dataset_dir_new), exist_ok=True)

for idx, class_dir in enumerate(class_dirs, 1):
    # Sprawdzanie czy klasa obrazów jest folderem
    class_dir_path = get_full_path(dataset_dir_current, class_dir)
    if not os.path.isdir(class_dir_path):
        continue

    # Tworzenie folderu wyjściowego dla klasy obrazów
    os.makedirs(get_full_path(dataset_dir_new, class_dir), exist_ok=True)

    # Zbieranie listy obrazów w klasie do wybranego limitu obrazów na klasę
    full_class_imgs = [image for image in os.listdir(class_dir_path) if os.path.splitext(image)[1].lower() in img_extentions]
    class_imgs = random.sample(full_class_imgs, min(img_class_limit, len(full_class_imgs)))
    num_of_imgs = len(class_imgs)

    # ETAP 1 - Zmiana nazwy obrazów i kopiowanie do nowego folderu
    for i, img_name in tqdm(enumerate(class_imgs), total=num_of_imgs, desc=f"Kopiowanie klasy {class_dir}", bar_format=bar_format, dynamic_ncols=True):
        current_img_path = get_full_path(class_dir_path, img_name)
        img_new_name = f"{class_dir}-{i+1}{os.path.splitext(img_name)[1].lower()}"
        new_img_path = get_full_path(dataset_dir_new, class_dir, img_new_name)
        shutil.copy(current_img_path, new_img_path)

    # ETAP 2 - Augmentacja obrazów w celu dopełnienia wybranego limitu obrazów na klasę
    missing_imgs = img_class_limit - num_of_imgs
    if (missing_imgs > 0):
        original_imgs = [get_full_path(dataset_dir_new, class_dir, f"{class_dir}-{i+1}{os.path.splitext(class_imgs[i])[1].lower()}") for i in range(num_of_imgs)]

        for i in tqdm(range(missing_imgs), desc=f"Augmentacja klasy {class_dir}", bar_format=bar_format, dynamic_ncols=True):
            img_path = original_imgs[i % num_of_imgs]
            image = imageio.v2.imread(img_path)
            augmented_image = augmenter(image=image)["image"]
            new_img_name = f"{class_dir}-{num_of_imgs + i + 1}.jpg"
            output_img_path = get_full_path(dataset_dir_new, class_dir, new_img_name)
            imageio.imwrite(output_img_path, augmented_image)

    # ETAP 3 - Skalowanie obrazów do wybranego rozmiaru
    new_class_dir_path = get_full_path(dataset_dir_new, class_dir)
    for img_file in tqdm(os.listdir(new_class_dir_path), desc=f"Skalowanie klasy {class_dir}", bar_format=bar_format, dynamic_ncols=True):
        img_path = os.path.join(new_class_dir_path, img_file)
        image = imageio.v2.imread(img_path)
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        imageio.imwrite(img_path, resized_image)

    print(f"[{idx}/{len(class_dirs)}] Klasa: {class_dir}, Skopiowane obrazy: {num_of_imgs}, Zaugmentowane obrazy: {missing_imgs}, Przeskalowane obrazy: {len(os.listdir(new_class_dir_path))}")
    print()

print(f"GOTOWE: Zakończono proces przenoszenia, augmentacji i skalowania obrazów. Każda klasa powinna zawierać 700 obrazów o rozmiarach {target_size[0]} x {target_size[1]}")
