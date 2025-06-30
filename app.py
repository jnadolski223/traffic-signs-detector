import cv2
import base64
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from keras.models import load_model
from keras.preprocessing.image import img_to_array

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

DETECTOR_MODEL_PATH = "detection-model.pt"
CLASSIFICATOR_MODEL_PATH = "classification-model.keras"
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

logging.info("Wczytywanie modelu detektora...")
detector_model = YOLO(DETECTOR_MODEL_PATH)
logging.info("Model detektora załadowany.")

logging.info("Wczytywanie modelu klasyfikatora...")
classificator_model = load_model(CLASSIFICATOR_MODEL_PATH)
logging.info("Model klasyfikatora załadowany.")


def analyze_image(image_bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = detector_model.predict(image, verbose=False)
    response_data = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            resized = cv2.resize(cropped, CLASSIFICATOR_IMAGE_FORMAT)
            input_img = img_to_array(resized)
            input_img = np.expand_dims(input_img, axis=0)

            prediction = classificator_model.predict(input_img, verbose=0)
            predicted_class = np.argmax(prediction)
            class_name = class_name = CLASS_NAMES.get(predicted_class, f"Znak: {predicted_class}")

            label = f"{class_name}, konf: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

            response_data.append({
                "class_id": int(predicted_class),
                "class_name": class_name,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })
    
    final_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", final_image)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return response_data, encoded_image

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    logging.info("Otrzymano żadanie POST /analyze")

    if "image" not in request.files:
        logging.warning("Żądanie nie zawiera pliku 'image'")
        return jsonify({ "error": "Brak pliku obrazu" }), 400
    
    file = request.files["image"]
    image_bytes = file.read()

    try:
        logging.info("Analizuję obraz...")
        result_data, image_b64 = analyze_image(image_bytes)
        logging.info("Zakończono analizę. Wykryto %d znaków", len(result_data))
        return jsonify({
            "detected_signs": result_data,
            "image_base64": image_b64
        })
    except Exception as e:
        logging.exception("Błąd podczas analizy obrazu:", e)
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    logging.info("Uruchamianie serwera Flask...")
    app.run(debug=True)