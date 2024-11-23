from PIL import Image, ImageOps
import os
import hashlib
import shutil
from pathlib import Path


def preprocess_images_with_padding(source_dir, output_dir, target_size, mode, clean_output, padding_color=(0, 0, 0)):
    """
    Preprocesează imaginile: redimensionare proporțională, completare cu padding și conversie.

    Args:
        source_dir (str): Directorul sursă cu imaginile originale.
        output_dir (str): Directorul destinație pentru imaginile preprocesate.
        target_size (tuple): Dimensiunile țintă pentru imagini (lățime, înălțime).
        mode (str): Formatul imaginii (e.g., "RGB", "L" pentru tonuri de gri).
        clean_output (bool): Dacă este True, șterge imaginile procesate anterior.
        padding_color (tuple): Culoarea folosită pentru completarea bordurilor (default: negru).
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Șterge directorul de ieșire dacă este necesar
    if clean_output:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cleaned output directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Hash pentru eliminarea duplicatelor
    seen_hashes = set()

    for image_path in source_dir.glob("*.*"):
        try:
            with Image.open(image_path) as img:
                # Conversie la modul specificat
                img = img.convert(mode)

                # Redimensionare proporțională
                img.thumbnail(target_size, Image.Resampling.LANCZOS)

                # Calcularea padding-ului
                width, height = img.size
                new_img = Image.new(mode, target_size, padding_color)
                left = (target_size[0] - width) // 2
                top = (target_size[1] - height) // 2
                new_img.paste(img, (left, top))

                # Calcularea hash-ului imaginii pentru eliminarea duplicatelor
                img_hash = hashlib.md5(new_img.tobytes()).hexdigest()
                if img_hash in seen_hashes:
                    print(f"Duplicate found and skipped: {image_path}")
                    continue
                seen_hashes.add(img_hash)

                # Salvare imagine procesată
                output_path = output_dir / image_path.name
                new_img.save(output_path)
                print(f"Processed and saved: {output_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def main():
    # Setări implicite pentru directoare
    default_source_directory = "./data_project/dataset/Custom"
    default_output_directory = "./data_project/preprocessed/Custom"

    # Solicitare date dinamic din partea utilizatorului
    source_directory = input(f"Enter the source directory for images (default: {default_source_directory}): ").strip()
    if not source_directory:
        source_directory = default_source_directory

    output_directory = input(
        f"Enter the output directory for preprocessed images (default: {default_output_directory}): ").strip()
    if not output_directory:
        output_directory = default_output_directory

    # Dimensiuni introduse dinamic
    try:
        width = int(input("Enter the target image width (e.g., 224): ").strip())
        height = int(input("Enter the target image height (e.g., 224): ").strip())
        target_size = (width, height)
    except ValueError:
        print("Invalid dimensions provided. Using default size: (224, 224).")
        target_size = (224, 224)

    # Alegerea formatului (RGB sau tonuri de gri)
    color_mode = input("Enter the color mode (RGB for color, L for grayscale): ").strip().upper()
    if color_mode not in ["RGB", "L"]:
        print("Invalid color mode provided. Using default: RGB.")
        color_mode = "RGB"

    # Întreabă utilizatorul dacă vrea să șteargă imaginile procesate anterior
    clean_output_choice = input(
        "Do you want to clean the processed images directory before running? (yes/no): ").strip().lower()
    clean_output = clean_output_choice == "yes"

    # Rularea funcției de preprocesare
    preprocess_images_with_padding(
        source_directory,
        output_directory,
        target_size,
        color_mode,
        clean_output
    )


if __name__ == "__main__":
    main()

