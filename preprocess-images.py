import os
import shutil
import random
import imageio
import albumentations as A
from tqdm import tqdm

def get_full_path(*path_args):
    return os.path.join(os.path.dirname(__file__), *path_args)

# Model do augmentacji zdjęć
augmenter = A.Compose([
    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.5),
    A.MotionBlur(p=0.2, blur_limit=3),
    A.CLAHE(p=0.3, clip_limit=(1, 4)),
    A.HueSaturationValue(p=0.3, hue_shift_limit=3, sat_shift_limit=10, val_shift_limit=10),
    A.MultiplicativeNoise(p=0.5, multiplier=(0.9, 1.1)),
])

# Główne zmienne
dataset_dir_current = "classification-dataset"
dataset_dir_new = "classification-dataset-fixed"
img_extentions = [".jpg", ".jpeg", ".png"]
img_class_limit = 700
bar_format = (
    "{desc}: |{bar}| {percentage:.1f}% "
    "[{n_fmt}/{total_fmt} obrazy przeanalizowane, "
    "Czas: {elapsed}, "
    "Pozostało: {remaining}, "
    "Prędkość: {rate_fmt}]"
)

# Sprawdzanie obecności folderu z datasetem
print(f"Sprawdzam folder {get_full_path(dataset_dir_current)}")
class_dirs = os.listdir(get_full_path(dataset_dir_current))
class_dirs_size = len(class_dirs)
print(f"Liczba klas obrazów: {class_dirs_size}")

# Tworzenie folderu wyjściowego dla nowego datasetu
print(f"Tworzenie nowego folderu o nazwie: {get_full_path(dataset_dir_new)}")
os.makedirs(get_full_path(dataset_dir_new), exist_ok=True)

help_iterator = 0
for class_dir in class_dirs:
    # Sprawdzanie czy klasa obrazów jest folderem
    class_dir_path = get_full_path(dataset_dir_current, class_dir)
    if not os.path.isdir(class_dir_path):
        continue

    # Tworzenie folderu wyjściowego dla klasy obrazów
    os.makedirs(get_full_path(dataset_dir_new, class_dir), exist_ok=True)

    # Zbieranie listy obrazów w klasie i wybieranie maksymalnie 700 z nich
    full_class_imgs = [image for image in os.listdir(class_dir_path) if os.path.splitext(image)[1].lower() in img_extentions]
    class_imgs = random.sample(full_class_imgs, min(img_class_limit, len(full_class_imgs)))
    num_of_imgs = len(class_imgs)

    # Kopiowanie istniejących obrazów z klasy do nowego folderu i zmiania ich nazwy
    for i, img_name in tqdm(enumerate(class_imgs), total=num_of_imgs, desc=f"Kopiowanie klasy {class_dir}", bar_format=bar_format, dynamic_ncols=True):
        current_img_path = get_full_path(class_dir_path, img_name)
        img_new_name = f"{class_dir}-{i+1}{os.path.splitext(img_name)[1].lower()}"
        new_img_path = get_full_path(dataset_dir_new, class_dir, img_new_name)
        shutil.copy(current_img_path, new_img_path)

    # Augmentacja obrazów w celu dopełnienia limitu obrazów
    missing_imgs = img_class_limit - num_of_imgs
    if (missing_imgs > 0):
        original_imgs = [get_full_path(dataset_dir_new, class_dir, f"{class_dir}-{i+1}{os.path.splitext(class_imgs[i])[1].lower()}") for i in range(num_of_imgs)]

        for i in tqdm(range(missing_imgs), desc=f"Augmentacja klasy {class_dir}", bar_format=bar_format, dynamic_ncols=True):
            img_path = original_imgs[i % num_of_imgs]
            image = imageio.v2.imread(img_path)

            # Zastosowanie augmentacji na obrazie
            augmented_image = augmenter(image=image)["image"]

            # Zapisywanie nowego obrazu
            new_img_name = f"{class_dir}-{num_of_imgs + i + 1}.jpg"
            output_img_path = get_full_path(dataset_dir_new, class_dir, new_img_name)
            imageio.imwrite(output_img_path, augmented_image)
    
    help_iterator += 1
    print(f"[{help_iterator}/{class_dirs_size}] Skopiowano {num_of_imgs} obrazów oraz augmentowano {missing_imgs} obrazów dla klasy {class_dir}")
    print()

print("GOTOWE: Zakończono proces przenoszenia i augmentacji obrazów. Każda klasa powinna zawierać 700 obrazów")