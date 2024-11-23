import os
import shutil
from pathlib import Path
import random


def clean_directory(directory_path):
    """
    Verifică dacă directorul există și are conținut. Întreabă utilizatorul dacă dorește să îl golească.
    Args:
        directory_path (Path): Calea către director.
    Returns:
        bool: True dacă directorul a fost curățat, False dacă fișierele existente sunt păstrate.
    """
    if directory_path.exists() and any(directory_path.iterdir()):
        choice = input(f"Directory {directory_path} is not empty. Do you want to clean it? (yes/no): ").strip().lower()
        if choice == "yes":
            shutil.rmtree(directory_path)
            directory_path.mkdir(parents=True, exist_ok=True)
            print(f"Cleaned directory: {directory_path}")
            return True
        elif choice == "no":
            print(f"Keeping existing files in directory: {directory_path}")
            return False
        else:
            print("Invalid input. Keeping existing files.")
            return False
    else:
        directory_path.mkdir(parents=True, exist_ok=True)
        return True


def split_dataset_single_class(source_dir, output_dir, dataset_name="Custom", train_ratio=0.7, validation_ratio=0.2,
                               test_ratio=0.1):
    """
    Împarte imaginile dintr-un singur director în seturi de antrenare, validare și testare.

    Args:
        source_dir (str): Directorul sursă cu imaginile preprocesate.
        output_dir (str): Directorul principal pentru seturile împărțite.
        dataset_name (str): Numele dataset-ului (e.g., "Custom").
        train_ratio (float): Proporția datelor pentru antrenare (default: 0.7).
        validation_ratio (float): Proporția datelor pentru validare (default: 0.2).
        test_ratio (float): Proporția datelor pentru testare (default: 0.1).
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Validare proporții
    if round(train_ratio + validation_ratio + test_ratio, 2) != 1.0:
        raise ValueError("Ratios must sum to 1. Current sum: "
                         f"{train_ratio + validation_ratio + test_ratio}")

    # Creează directoare pentru train, validation și test
    dataset_dirs = {
        "train": output_dir / "train" / dataset_name,
        "validation": output_dir / "validation" / dataset_name,
        "test": output_dir / "test" / dataset_name,
    }

    # Verifică și gestionează directoarele
    for split, dir_path in dataset_dirs.items():
        clean_directory(dir_path)

    # Obține toate imaginile din directorul sursă
    images = list(source_dir.glob("*.*"))
    if not images:
        raise ValueError(f"No images found in source directory: {source_dir}")

    random.shuffle(images)

    # Separă datele
    train_split = int(len(images) * train_ratio)
    validation_split = int(len(images) * (train_ratio + validation_ratio))

    train_images = images[:train_split]
    validation_images = images[train_split:validation_split]
    test_images = images[validation_split:]

    # Copiază imaginile în directoarele corespunzătoare, evitând duplicatele
    for split, image_list in zip(["train", "validation", "test"], [train_images, validation_images, test_images]):
        for image in image_list:
            destination_path = dataset_dirs[split] / image.name
            if not destination_path.exists():  # Evită duplicatele
                shutil.copy(image, destination_path)

    print(f"Processed dataset '{dataset_name}': "
          f"{len(train_images)} train, {len(validation_images)} validation, {len(test_images)} test")


if __name__ == "__main__":
    # Setări implicite pentru directoare și dataset
    source_directory = input(
        "Enter the source directory for images (default: ./data_project/preprocessed/Custom): ").strip()
    if not source_directory:
        source_directory = "./data_project/preprocessed/Custom"

    output_directory = input(
        "Enter the output directory for split datasets (default: ./data_project/dataset_split): ").strip()
    if not output_directory:
        output_directory = "./data_project/dataset_split"

    dataset_name = input("Enter the name of the dataset (default: Custom): ").strip()
    if not dataset_name:
        dataset_name = "Custom"

    # Introducere dinamică pentru proporții
    try:
        train_ratio = input("Enter the train ratio (default: 0.7): ").strip()
        validation_ratio = input("Enter the validation ratio (default: 0.2): ").strip()
        test_ratio = input("Enter the test ratio (default: 0.1): ").strip()

        # Folosim valorile implicite dacă utilizatorul nu introduce nimic
        train_ratio = float(train_ratio) if train_ratio else 0.7
        validation_ratio = float(validation_ratio) if validation_ratio else 0.2
        test_ratio = float(test_ratio) if test_ratio else 0.1

        # Validare proporții
        if round(train_ratio + validation_ratio + test_ratio, 2) != 1.0:
            raise ValueError("The provided ratios must sum to 1.")

    except ValueError as e:
        print(f"Invalid input for ratios: {e}")
        print("Using default ratios: 0.7 train, 0.2 validation, 0.1 test.")
        train_ratio, validation_ratio, test_ratio = 0.7, 0.2, 0.1

    # Apel funcție cu proporțiile introduse
    try:
        split_dataset_single_class(source_directory, output_directory, dataset_name, train_ratio, validation_ratio,
                                   test_ratio)
    except Exception as e:
        print(f"Error: {e}")
