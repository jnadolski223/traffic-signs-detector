import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Główne zmienne
DETECTOR_MODEL_PATH = "detection-model.pt"
CLASSIFICATOR_MODEL_PATH = "classification-model.keras"
TEST_DATASET_DIR = "test-detection-dataset"
TEST_RESULTS_DIR = "test-results"
PROGRESSBAR_FORMAT = (
    "{desc}: |{bar}| {percentage:.1f}% "
    "[Przeanalizowano: {n_fmt}/{total_fmt}, Czas: {elapsed}, Pozostało: {remaining}]"
)
CLASSIFICATOR_IMAGE_FORMAT = (224, 224)
CLASS_NAMES = {
    0: "A1", 1: "A11a", 2: "A12", 3: "A16", 4: "A17",
    5: "A18b", 6: "A2", 7: "A21", 8: "A29", 9: "A3",
    10: "A30", 11: "A32", 12: "A4", 13: "A6a", 14: "A6b",
    15: "A6c", 16: "A6d", 17: "A6e", 18: "A7", 19: "B1",
    20: "B2", 21: "B20", 22: "B21", 23: "B22", 24: "B23",
    25: "B25", 26: "B26", 27: "B33", 28: "B36", 29: "B41",
    30: "B5", 31: "C10", 32: "C12", 33: "C2", 34: "C4",
    35: "C5", 36: "C9", 37: "D1", 38: "D15", 39: "D18",
    40: "D2", 41: "D23", 42: "D28", 43: "D29", 44: "D3",
    45: "D42", 46: "D43", 47: "D4a", 48: "D6", 49: "D6b"
}

detector_model = YOLO(DETECTOR_MODEL_PATH)
classificator_model = load_model(CLASSIFICATOR_MODEL_PATH)

def process_image(image_path, savefile):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Nie można wczytać obrazu: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = detector_model.predict(image, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            resized = cv2.resize(cropped, CLASSIFICATOR_IMAGE_FORMAT)
            input_img = img_to_array(resized)
            input_img = np.expand_dims(input_img, axis=0)

            prediction = classificator_model.predict(input_img, verbose=0)
            predicted_class = np.argmax(prediction)

            class_name = CLASS_NAMES.get(predicted_class, f"Znak: {predicted_class}")
            label = f"{class_name}, konf: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(savefile, image)

def main():
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    image_files = [f for f in os.listdir(TEST_DATASET_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for idx, image_file in enumerate(tqdm(image_files, desc="Przetwarzanie obrazów", bar_format=PROGRESSBAR_FORMAT, dynamic_ncols=True)):
        input_path = os.path.join(TEST_DATASET_DIR, image_file)
        output_path = os.path.join(TEST_RESULTS_DIR, f"result-{idx+1}.jpg")
        process_image(input_path, output_path)

if __name__ == "__main__":
    main()