import shutil
from pathlib import Path
import random
import zipfile
import tarfile
import kagglehub


def create_directory_structure(base_path):
    raw_data_dir = Path(base_path) / "raw_data_sets"
    processed_data_dir = Path(base_path) / "dataset"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    return raw_data_dir, processed_data_dir


def select_and_copy_images(source_files, destination_dir, max_images):
    """
    Selectează și copiază un număr specific de imagini în directorul destinație.
    """
    random.shuffle(source_files)
    selected_files = source_files[:max_images]

    destination_dir.mkdir(parents=True, exist_ok=True)

    for file in selected_files:
        destination_path = destination_dir / file.name
        shutil.copy(file, destination_path)

    print(f"{len(selected_files)} images copied to {destination_dir}.")


def extract_and_select_images(archive_path, destination_dir, max_images, archive_type="zip"):
    """
    Extragerea unui subset de imagini din arhive ZIP sau TGZ.
    """
    archive_path = Path(archive_path)
    destination_dir = Path(destination_dir)

    if not archive_path.exists():
        print(f"Error: {archive_type.upper()} file {archive_path} does not exist.")
        return

    print(f"Opening {archive_type.upper()} file {archive_path}...")
    extracted_files = []

    if archive_type == "zip":
        with zipfile.ZipFile(archive_path, 'r') as archive:
            all_files = [f for f in archive.namelist() if f.endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(all_files)
            selected_files = all_files[:max_images]

            for file in selected_files:
                file_name = Path(file).name
                with archive.open(file) as source, open(destination_dir / file_name, 'wb') as target:
                    shutil.copyfileobj(source, target)
                extracted_files.append(destination_dir / file_name)
    elif archive_type == "tgz":
        with tarfile.open(archive_path, 'r') as archive:
            all_files = [member for member in archive.getmembers() if member.name.endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(all_files)
            selected_files = all_files[:max_images]

            for member in selected_files:
                member.name = Path(member.name).name
                archive.extract(member, path=destination_dir)
                extracted_files.append(destination_dir / member.name)

    print(f"{len(extracted_files)} images extracted to {destination_dir}.")


def download_and_select_kaggle_dataset(dataset_name, destination_dir, max_images):
    """
    Descărcare și selecție a imaginilor dintr-un set de date Kaggle.
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Kaggle dataset {dataset_name}...")
    raw_dir = kagglehub.dataset_download(dataset_name)

    if not raw_dir:
        print(f"Error: Failed to download Kaggle dataset {dataset_name}.")
        return

    images = [f for f in Path(raw_dir).glob("**/*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    select_and_copy_images(images, destination_dir, max_images)


def manage_dataset_data(dataset_name, dataset_dir):
    """
    Gestionarea directoarelor din `dataset`.
    """
    dataset_dir = Path(dataset_dir) / dataset_name

    if not dataset_dir.exists() or not any(dataset_dir.glob("**/*")):
        print(f"Directory {dataset_dir} does not exist or is empty. It will be created and populated with new data.")
        return "clean"

    images = [f for f in dataset_dir.glob("**/*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not images:
        print(f"Directory {dataset_dir} contains no valid images. It will be populated with new data.")
        return "clean"

    choice = input(f"Directory {dataset_dir} contains images. Do you want to:\n"
                   "1. Keep existing images and add new ones (type 'keep')\n"
                   "2. Remove existing images and replace with new ones (type 'clean')\n"
                   "Choice: ").strip().lower()
    return choice


def load_dataset(dataset_choice, base_path, max_images=500):
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

    dataset_dir = processed_data_dir / dataset_choice
    user_choice = manage_dataset_data(dataset_choice, processed_data_dir)

    if user_choice == "clean":
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

    if dataset_choice == "CelebA":
        extract_and_select_images(datasets["CelebA"], dataset_dir, max_images, archive_type="zip")
    elif dataset_choice == "LFW":
        extract_and_select_images(datasets["LFW"], dataset_dir, max_images, archive_type="tgz")
    elif dataset_choice == "FER-2013":
        download_and_select_kaggle_dataset("msambare/fer2013", dataset_dir, max_images)
    elif dataset_choice == "Custom":
        custom_images_dir = datasets["Custom"]
        images = [f for f in Path(custom_images_dir).glob("**/*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        select_and_copy_images(images, dataset_dir, max_images)


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
