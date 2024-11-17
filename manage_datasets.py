import os
import shutil
from pathlib import Path
import random
import zipfile
import tarfile
import requests
import kagglehub


def create_directory_structure(base_path):
    raw_data_dir = Path(base_path) / "raw_data_sets"
    processed_data_dir = Path(base_path) / "dataset"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    return raw_data_dir, processed_data_dir


def clean_directory(directory_path, prompt=True):
    relative_path = directory_path.relative_to(Path.cwd())
    if directory_path.exists() and any(directory_path.iterdir()):
        if prompt:
            choice = input(f"Directory {relative_path} is not empty. Do you want to clean it? (yes/no): ").lower()
        else:
            choice = 'yes'
        if choice == 'yes':
            shutil.rmtree(directory_path)
            directory_path.mkdir(parents=True, exist_ok=True)
            print(f"Directory {relative_path} cleaned.")
        else:
            print("Directory cleaning skipped.")


def select_images(source_dir, destination_dir, max_images=500):
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    if not source_dir.exists():
        print(f"Source directory {source_dir} does not exist.")
        return

    images = [f for f in source_dir.glob("**/*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    random.shuffle(images)

    if len(images) < max_images:
        print(f"Warning: Only {len(images)} images available in {source_dir}, but {max_images} requested.")
    images = images[:max_images]

    destination_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        destination_path = destination_dir / image.name
        try:
            shutil.copy(image, destination_path)
        except Exception as e:
            print(f"Error copying {image}: {e}")

    print(f"{len(images)} random images copied to {destination_dir}.")


def extract_zip_and_select_images(zip_path, destination_dir, max_images):
    zip_path = Path(zip_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Error: ZIP file {zip_path} does not exist.")
        return

    print(f"Opening ZIP file {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = [f for f in zip_ref.namelist() if f.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(all_files)

        if len(all_files) < max_images:
            print(f"Warning: Only {len(all_files)} images available in {zip_path}, but {max_images} requested.")
            max_images = len(all_files)

        selected_files = all_files[:max_images]

        print(f"Extracting {len(selected_files)} images to {destination_dir}...")
        for file in selected_files:
            extracted_path = destination_dir / Path(file).name
            with zip_ref.open(file) as source, open(extracted_path, 'wb') as target:
                shutil.copyfileobj(source, target)

    print(f"{len(selected_files)} images extracted to {destination_dir}.")


def extract_tgz_and_select_images(tgz_path, destination_dir, max_images):
    tgz_path = Path(tgz_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if not tgz_path.exists():
        print(f"Error: TGZ file {tgz_path} does not exist.")
        return

    print(f"Opening TGZ file {tgz_path}...")
    with tarfile.open(tgz_path, 'r') as tar_ref:
        all_files = [member for member in tar_ref.getmembers() if member.name.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(all_files)

        if len(all_files) < max_images:
            print(f"Warning: Only {len(all_files)} images available in {tgz_path}, but {max_images} requested.")
            max_images = len(all_files)

        selected_files = all_files[:max_images]

        print(f"Extracting {len(selected_files)} images to {destination_dir}...")
        for member in selected_files:
            member.name = Path(member.name).name
            tar_ref.extract(member, path=destination_dir)

    print(f"{len(selected_files)} images extracted to {destination_dir}.")


def download_kaggle_dataset_and_select_images(dataset_name, destination_dir, max_images):
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Kaggle dataset {dataset_name}...")
    # Descărcarea setului de date fără `target_path`
    raw_dir = kagglehub.dataset_download(dataset_name)

    if not raw_dir:
        print(f"Error: Failed to download Kaggle dataset {dataset_name}.")
        return

    # Obține lista tuturor imaginilor descărcate
    images = [f for f in Path(raw_dir).glob("**/*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    random.shuffle(images)

    if len(images) < max_images:
        print(f"Warning: Only {len(images)} images available in {dataset_name}, but {max_images} requested.")
        max_images = len(images)

    selected_images = images[:max_images]

    print(f"Copying {len(selected_images)} images to {destination_dir}...")
    for image in selected_images:
        shutil.copy(image, destination_dir / image.name)

    print(f"{len(selected_images)} images copied to {destination_dir}.")


def manage_raw_data(dataset_name, raw_data_dir):
    raw_dir = raw_data_dir / dataset_name
    cwd = Path.cwd()

    try:
        relative_path = raw_dir.relative_to(cwd)
    except ValueError:
        relative_path = raw_dir

    if not raw_dir.exists() or not any(raw_dir.glob("**/*")):
        print(f"Directory {relative_path} is empty. Fetching new data...")
        return True

    images = [f for f in raw_dir.glob("**/*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not images:
        print(f"Directory {relative_path} contains no images. Fetching new data...")
        return True

    choice = input(f"Directory {relative_path} contains images. Do you want to update? (yes/no): ").lower()
    return choice == 'yes'


def load_dataset(dataset_choice, base_path, max_images=500, clean_existing=True):
    raw_data_dir, processed_data_dir = create_directory_structure(base_path)

    datasets = {
        "CelebA": raw_data_dir / "img_align_celeba.zip",
        "LFW": raw_data_dir / "lfw.tgz",
        "FER-2013": "msambare/fer2013",
        "Custom": raw_data_dir / "custom_images"
    }

    if dataset_choice not in datasets:
        print(f"Invalid dataset choice: {dataset_choice}.")
        return

    if dataset_choice == "CelebA":
        zip_path = datasets["CelebA"]
        if manage_raw_data("img_align_celeba", raw_data_dir):
            extract_zip_and_select_images(zip_path, processed_data_dir / dataset_choice, max_images)
    elif dataset_choice == "LFW":
        tgz_path = datasets["LFW"]
        if manage_raw_data("lfw", raw_data_dir):
            extract_tgz_and_select_images(tgz_path, processed_data_dir / dataset_choice, max_images)
    elif dataset_choice == "FER-2013":
        if manage_raw_data("msambare_fer2013", raw_data_dir):
            download_kaggle_dataset_and_select_images("msambare/fer2013", processed_data_dir / dataset_choice,
                                                      max_images)
    elif dataset_choice == "Custom":
        select_images(datasets["Custom"], processed_data_dir / dataset_choice, max_images)


if __name__ == "__main__":
    base_path = "./data_project"

    while True:
        dataset_name = input("Enter dataset to use (CelebA, LFW, FER-2013, Custom): ").strip()
        if dataset_name not in ["CelebA", "LFW", "FER-2013", "Custom"]:
            print("Invalid dataset choice. Please select from: CelebA, LFW, FER-2013, Custom.")
            continue

        try:
            max_images_to_select = int(input("Enter the number of images to select: ").strip())
        except ValueError:
            print("Please enter a valid number for the maximum images.")
            continue

        load_dataset(dataset_name, base_path, max_images_to_select)

        next_action = input("Do you want to process another dataset? (yes/no): ").strip().lower()
        if next_action == "no":
            print("Exiting the program. Goodbye!")
            break
        elif next_action != "yes":
            print("Invalid input. Exiting the program.")
            break
