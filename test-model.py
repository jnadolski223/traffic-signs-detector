import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import models, utils
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model-cnn-2025-06-27.keras")
TEST_DIR = os.path.join(os.path.dirname(__file__), "split-classification-dataset/test")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
PROGRESSBAR_FORMAT = (
    "{desc}: |{bar}| {percentage:.1f}% "
    "[Przeanalizowano: {n_fmt}/{total_fmt}, Czas: {elapsed}, Pozostało: {remaining}]"
)

# Funckja wczytuje dane testowe z datasetu i go zwraca razem z nazwami klas
def load_test_data():
    test_set = utils.image_dataset_from_directory(
        directory=TEST_DIR,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False
    )
    class_names = test_set.class_names
    return test_set, class_names

# Funkcja testuje podany model i wyświetla macierz błędów oraz inne statystyki
def compute_confusion_matrix(model, test_set, class_names):
    y_true = []
    y_pred = []

    for batch_images, batch_labels in tqdm(test_set, desc="Predicting", bar_format=PROGRESSBAR_FORMAT, dynamic_ncols=True):
        preds = model.predict(batch_images, verbose=0)
        y_true.extend(batch_labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n========== Macierz błędów ==========")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(13, 13))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Przewidywana klasa")
    plt.ylabel("Prawdziwa klasa")
    plt.title("Macierz błędów")
    plt.show()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")

    print("\n========== Raport klasyfikacji ==========")
    print(classification_report(y_true, y_pred, target_names=class_names))

    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                errors.append((class_names[i], class_names[j], cm[i, j]))
    errors.sort(key=lambda x: -x[2])

    print("\nTop 10 największych pomyłek:")
    for true_label, pred_label, count in errors[:10]:
        print(f"{true_label} -> {pred_label}: {count}")

# Funkcja wczytuje model, testuje go i wyświetla statystyki
def main():
    model = models.load_model(MODEL_PATH)
    print(f"Model załadowany z pliku {MODEL_PATH}")

    test_set, class_names = load_test_data()

    compute_confusion_matrix(model, test_set, class_names)

if __name__ == "__main__":
    main()