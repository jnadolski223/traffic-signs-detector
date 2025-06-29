import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import models, utils
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Główne zmienne
MODEL_PATH = os.path.join("classification-model-results", "classification-model.keras")
TEST_DIR = os.path.join("split-classification-dataset", "test")

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

# Funkcja testuje podany model i wyświetla raport klasyfikacji, macierz błędów oraz wykres najczęstszych błędów
def evaluate_model(model, test_set, class_names):
    y_true = []
    y_pred = []

    for batch_images, batch_labels in tqdm(test_set, desc="Predykowanie klas", bar_format=PROGRESSBAR_FORMAT, dynamic_ncols=True):
        preds = model.predict(batch_images, verbose=0)
        y_true.extend(batch_labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n========== Raport klasyfikacji ==========")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(13, 13))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig("classification-model-confusion-matrix.png", dpi=300)
    plt.show()
    plt.close()
    print("Macierz błędów zapisana jako classification-model-confusion-matrix.png")

    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                errors.append((class_names[i], class_names[j], cm[i, j]))
    errors.sort(key=lambda x: -x[2])
    top_errors = errors[:10]
    
    labels = [f"{t}→{p}" for t, p, _ in top_errors]
    values = [c for _, _, c in top_errors]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=values, y=labels, hue=labels, palette="Reds_r")
    plt.xlabel("Number of errors")
    plt.ylabel("Error (True → Predicted)")
    plt.title("Top 10 most common errors")
    plt.tight_layout()
    plt.savefig("classification-model-top-10-mistakes.png", dpi=300)
    plt.show()
    plt.close()
    print("Wykres top 10 pomyłek zapisany jako classification-model-top-10-mistakes.png")

# Funkcja wczytuje model, testuje go i wyświetla statystyki
def main():
    model = models.load_model(MODEL_PATH)
    print(f"Model załadowany z pliku {MODEL_PATH}")

    test_set, class_names = load_test_data()

    evaluate_model(model, test_set, class_names)

if __name__ == "__main__":
    main()