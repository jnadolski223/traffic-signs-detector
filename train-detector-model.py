from ultralytics import YOLO

# Główne zmienne
DATA_CONFIG_FILE = "detection-dataset.yaml"
EPOCHS = 50
PROJECT_DIR = "yolo-models"
OUTPUT_DIR = "road-sign-detector"

# Funkcja pobiera model YOLOv8n i dotrenowuje go na danych we wskazanym pliku
def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_CONFIG_FILE,
        epochs=EPOCHS,
        project=PROJECT_DIR,
        name=OUTPUT_DIR
    )

if __name__ == "__main__":
    main()