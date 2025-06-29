import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# Główne zmienne
VAL_DIR = os.path.join("split-detection-dataset", "images", "val")
RESULTS_DIR = "evaluated-images"
PROGRESSBAR_FORMAT = (
    "{desc}: |{bar}| {percentage:.1f}% "
    "[Przeanalizowano: {n_fmt}/{total_fmt}, Czas: {elapsed}, Pozostało: {remaining}]"
)

# Funckja wczytuje model, testuje go i zapisuje wyniki
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    images = [os.path.join(VAL_DIR, f) for f in os.listdir(VAL_DIR)]

    model = YOLO("detection-model-results/weights/best.pt")
    for image in tqdm(images, desc="Predykcja YOLO", bar_format=PROGRESSBAR_FORMAT, dynamic_ncols=True):
        results = model(image, verbose=False)
        result = results[0]
        rendered = result.plot()
        cv2.imwrite(os.path.join(RESULTS_DIR, os.path.basename(image)), rendered)

if __name__ == "__main__":
    main()