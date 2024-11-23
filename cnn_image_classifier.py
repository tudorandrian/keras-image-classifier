import os
import shutil
from pathlib import Path
import hashlib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint

# Configurări de bază
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ascundem mesajele TensorFlow despre optimizare CPU
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 20
DATASET_PATH = "./data_project/dataset_split"
RESULTS_PATH = "./data_project/models_results"
MODELS_PATH = "./data_project/models"

# Maparea claselor (limba maternă -> engleză)
class_mapping = {
    "Oameni": "Humans",
    "Caini": "Dogs",
    "Pisici": "Cats",
    "Cai": "Horses",
    "Cladiri": "Buildings"
}


# Funcție pentru crearea modelului CNN
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')  # Clasificare binară
    ])
    return model


# Funcție pentru antrenarea modelului
def train_model(selected_class):
    class_english = class_mapping[selected_class]
    train_dir = f"{DATASET_PATH}/train/Custom"
    val_dir = f"{DATASET_PATH}/validation/Custom"

    # Încărcăm datele de antrenament și validare
    train_dataset = image_dataset_from_directory(
        directory=train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    val_dataset = image_dataset_from_directory(
        directory=val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Creăm și compilăm modelul
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Configurăm salvarea modelului antrenat
    model_file = f"{MODELS_PATH}/{class_english}.keras"
    checkpoint = ModelCheckpoint(model_file, save_best_only=True, monitor='val_accuracy', mode='max')

    # Antrenăm modelul
    print(f"Training model for class '{selected_class}'...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )
    print(f"Model for class '{selected_class}' has been trained and saved at {model_file}.")


# Funcție pentru rularea modelului pe o singură clasă
def process_class(selected_class):
    # Verificăm dacă clasa selectată este validă
    if selected_class not in class_mapping:
        print(f"Invalid class. Choose from: {', '.join(class_mapping.keys())}")
        return

    class_english = class_mapping[selected_class]
    model_file = f"{MODELS_PATH}/{class_english}.keras"
    if not os.path.exists(model_file):
        print(f"Trained model for class '{selected_class}' not found. Please train it first.")
        train_now = input("Do you want to train it now? (yes/no): ").strip().lower()
        if train_now == "yes":
            train_model(selected_class)
        else:
            return

    # Încărcăm modelul
    model = tf.keras.models.load_model(model_file)
    print(f"Loaded model for class '{selected_class}'.")

    # Setăm calea pentru rezultate
    class_results_dir = Path(f"{RESULTS_PATH}/CNN_Custom/{selected_class}")
    class_results_dir.mkdir(parents=True, exist_ok=True)

    # Încărcăm dataset-ul de test
    test_dataset = image_dataset_from_directory(
        directory=f"{DATASET_PATH}/test/Custom",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Procesăm imaginile din setul de test
    for batch in test_dataset:
        images, labels = batch
        predictions = model.predict(images)

        for i, prediction in enumerate(predictions):
            if prediction.argmax() == 1:  # Dacă imaginea este clasificată pozitiv pentru clasa selectată
                image_path = f"{class_results_dir}/image_{i}.jpg"
                tf.keras.preprocessing.image.save_img(image_path, images[i])
                print(f"Saved image for class '{selected_class}': {image_path}")


# Exemplu de utilizare
if __name__ == "__main__":
    print("Available actions: run_model, create_class, delete_class")
    action = input("Enter the action you want to perform: ").strip()

    if action == "run_model":
        selected_class = input("Enter the name of the class to process (e.g., Oameni): ").strip()
        process_class(selected_class)
