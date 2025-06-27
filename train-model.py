import os
from datetime import datetime
from keras import layers, models, utils, callbacks
import matplotlib.pyplot as plt
import pandas as pd

DATASET_PATH = os.path.join(os.path.dirname(__file__), "split-classification-dataset")
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")
TEST_DIR = os.path.join(DATASET_PATH, "test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INPUT_SHAPE = (224, 224, 3)
NUMBER_OF_CLASSES = 50
EPOCHS = 100

# Funkcja tworzy i kompiluje model
def create_model():
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUMBER_OF_CLASSES, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

# Funkcja wyświetla statystyki uczenia się modelu
def summarize_diagnostics(history, file_name = "train_plot.png"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="Train accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="Validation accuracy")
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, color="gray")

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="Train loss")
    plt.plot(history.history.get("val_loss", []), label="Validation loss")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, color="gray")

    plt.tight_layout()

    plt.savefig(file_name)
    plt.show()

    print(f"Wykres zapisano jako: {os.path.abspath(file_name)}")

# Funkcja uczy utworzony model, a na końcu go zapisuje i wyświetla podsumowanie
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(os.path.join(os.path.dirname(__file__), "checkpoints"))

    final_model_name = f"model_cnn_final_{timestamp}.keras"
    checkpoint_filename = f"checkpoints/model_checkpoint_best_{timestamp}.keras"
    plot_filename = f"training_plot_{timestamp}.png"

    model = create_model()

    train_set = utils.image_dataset_from_directory(
        directory=TRAIN_DIR, labels="inferred", label_mode="int",
        batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=True
    )

    val_set = utils.image_dataset_from_directory(
        directory=VAL_DIR, labels="inferred",label_mode="int",
        batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=False
    )

    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model_checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_filename, monitor="val_accuracy", save_best_only=True)

    history = model.fit(train_set, validation_data=val_set, epochs=EPOCHS, callbacks=[early_stop, model_checkpoint])

    model.save(final_model_name)
    print(f"Model zapisano jako: {os.path.abspath(final_model_name)}")

    summarize_diagnostics(history, file_name=plot_filename)
    pd.DataFrame(history.history).to_csv(f"training_history_{timestamp}.csv")

if __name__ == "__main__":
    main()