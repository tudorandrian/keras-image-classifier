import os
import shutil
import hashlib
import uuid
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint

# Configurări de bază
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 20
DATASET_PATH = "./data_project/dataset_split"
RESULTS_PATH = "./data_project/models_results"
MODELS_PATH = "./data_project/models"

# Clase implicte: Oameni, Animale, Vehicule

# Funcție pentru validarea structurii directoarelor
def validate_structure():
    """
    Verifică dacă structura de directoare necesară este prezentă.
    """
    print("Validating dataset structure...")
    required_dirs = ["train", "validation", "test"]
    for split in required_dirs:
        split_path = Path(f"{DATASET_PATH}/{split}/Custom")
        if not split_path.exists():
            print(f"Error: Required directory '{split_path}' is missing. Please process the dataset.")
            return False
    print("Dataset structure is valid.")
    return True

# Funcție pentru crearea modelului CNN
def create_model(num_classes):
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
        layers.Dense(num_classes, activation='softmax')  # Clasificare multi-clasă
    ])
    return model

# Funcție pentru antrenarea modelului
def train_model(selected_class):
    print(f"Starting training for class '{selected_class}'...")

    # Setăm directoarele pentru train și validation
    train_dir = f"{DATASET_PATH}/train/Custom"
    val_dir = f"{DATASET_PATH}/validation/Custom"

    # Încărcăm datele de antrenament și validare
    train_dataset = image_dataset_from_directory(
        directory=train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=[selected_class]  # Specificăm clasa dorită
    )
    val_dataset = image_dataset_from_directory(
        directory=val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=[selected_class]  # Specificăm clasa dorită
    )

    # Calculăm numărul de clase
    num_classes = len(train_dataset.class_names)
    print(f"Number of classes detected: {num_classes}")

    # Creăm și compilăm modelul
    model = create_model(num_classes)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Configurăm salvarea modelului antrenat
    model_file = f"{MODELS_PATH}/{selected_class}.keras"
    checkpoint = ModelCheckpoint(model_file, save_best_only=True, monitor='val_accuracy', mode='max')

    # Antrenăm modelul
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )
    print(f"Model for class '{selected_class}' has been trained and saved at {model_file}.")


# Funcție pentru rularea modelului pe o singură clasă
def process_class(selected_class):
    print(f"Processing images for class '{selected_class}'...")
    model_file = f"{MODELS_PATH}/{selected_class}.keras"

    # Verificăm dacă modelul există
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

    # Evităm duplicatele folosind hash-uri
    saved_hashes = set(
        hashlib.md5(open(str(image_path), 'rb').read()).hexdigest()
        for image_path in class_results_dir.glob("*.jpg")
    )

    # Încărcăm dataset-ul de test
    test_dir = f"{DATASET_PATH}/test/Custom"
    test_dataset = image_dataset_from_directory(
        directory=test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Procesăm imaginile din setul de test
    for batch_index, batch in enumerate(test_dataset):
        images, labels = batch
        predictions = model.predict(images)

        for i, prediction in enumerate(predictions):
            if prediction.argmax() == 1:  # Dacă imaginea este clasificată pozitiv pentru clasa selectată
                print(f"Dacă imaginea este clasificată pozitiv pentru clasa selectată.")
                # Generăm un nume unic pentru imagine
                image_hash = hashlib.md5(images[i].numpy().tobytes()).hexdigest()
                if image_hash not in saved_hashes:  # Salvăm doar imaginile unice
                    unique_name = f"image_{batch_index}_{uuid.uuid4().hex[:8]}.jpg"
                    image_path = class_results_dir / unique_name
                    tf.keras.preprocessing.image.save_img(str(image_path), images[i])
                    saved_hashes.add(image_hash)
                    print(f"Saved image for class '{selected_class}': {image_path}")
                else:
                    print(f"Duplicate image skipped for class '{selected_class}'.")
            # else:
                # print(f"Imaginea NU este clasificată pozitiv pentru clasa selectată")
    print("All test images processed. End of sequence.")

# Funcția principală
def main():
    if not validate_structure():
        return

    while True:
        print("\nAvailable actions: train_model, run_model")
        action = input("Enter the action you want to perform: ").strip()

        if action == "train_model":
            selected_class = input("Enter the name of the class to train (e.g., Oameni): ").strip()
            train_model(selected_class)
        elif action == "run_model":
            selected_class = input("Enter the name of the class to process (e.g., Oameni): ").strip()
            process_class(selected_class)
        else:
            print("Invalid action. Please choose 'train_model' or 'run_model'.")

        continue_choice = input("\nDo you want to perform another action? (yes/no): ").strip().lower()
        if continue_choice != "yes":
            print("Exiting. Goodbye!")
            break

# Exemplu de utilizare
if __name__ == "__main__":
    main()
