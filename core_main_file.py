import os
print(os.getenv('TF_ENABLE_ONEDNN_OPTS'))  # Ar trebui să afișeze '0'

import manage_datasets
import image_preprocessing
import split_dataset_single_class
import cnn_image_classifier

# Lista fișierelor disponibile și funcțiile asociate
files = {
    "1": ("manage_datasets.py", manage_datasets.main),
    "2": ("image_preprocessing.py", image_preprocessing.main),
    "3": ("split_dataset_single_class.py", split_dataset_single_class.main),
    "4": ("cnn_image_classifier.py", cnn_image_classifier.main)
}

"""
Exemplu de Utilizare și Rulare din Consolă pentru fiecare fișier:

1. manage_datasets.py:
   - Utilizare:
     Acest script gestionează seturile de date. Permite descărcarea, extragerea și organizarea imaginilor din surse predefinite sau personalizate.
   - Rulare:
     python manage_datasets.py
     Exemplu interacțiune în consolă:
     - Enter dataset to use (CelebA, LFW, FER-2013, Custom): Custom
     - Enter the number of images to select: 50
     - Directory data_project\dataset\Custom contains images. Do you want to:
        1. Keep existing images and add new ones (type 'keep')
        2. Remove existing images and replace with new ones (type 'clean')
     - Choice: clean
        50 images copied to data_project\dataset\Custom.
     - Do you want to process another dataset? (yes/no): no
        Exiting the program. Goodbye!

2. image_preprocessing.py:
   - Utilizare:
     Preprocesează imaginile dintr-un director. Transformările includ redimensionarea imaginilor la dimensiuni standard, conversia la RGB/Grayscale și eliminarea duplicatelor.
   - Rulare:
     python image_preprocessing.py
     Exemplu interacțiune în consolă:
     - Enter the source directory for images (default: ./data_project/dataset/Custom):
     - Enter the output directory for processed images (default: ./data_project/preprocessed/Custom):
     - Enter image width (default: 224):
     - Enter image height (default: 224):

3. split_dataset_single_class.py:
   - Utilizare:
     Împarte imaginile dintr-un director sursă într-o structură organizată (train, validation, test) pe clase. Permite utilizatorului să specifice proporțiile seturilor.
   - Rulare:
     python split_dataset_single_class.py
     Exemplu interacțiune în consolă:
     - Enter the source directory for images (default: ./data_project/preprocessed/Custom):
     - Enter the output directory for split datasets (default: ./data_project/dataset_split):
     - Enter the name of the dataset (default: Custom):
     - Enter class names separated by commas (e.g., Oameni, Caini, Pisici, Cai, Cladiri):
     - Enter the train ratio (default: 0.7):
     - Enter the validation ratio (default: 0.2):
     - Enter the test ratio (default: 0.1):

4. cnn_image_classifier.py:
   - Utilizare:
     Rulează un model CNN pentru clasificarea imaginilor în funcție de clase. Include opțiuni pentru a antrena modele noi, a încărca modele existente și a salva rezultatele.
   - Rulare:
     python cnn_image_classifier.py
     Exemplu interacțiune în consolă:
     - Enter the action you want to perform (run_model, create_class, delete_class):
       Exemplu:
       - run_model: Rulează modelul CNN pentru o clasă selectată.
         Enter the name of the class to process (e.g., Oameni):
       - create_class: Creează un director pentru o nouă clasă.
         Enter the name of the class to manage (e.g., Oameni):
       - delete_class: Șterge toate rezultatele pentru o clasă specificată.
         Enter the name of the class to manage (e.g., Oameni):
"""


def print_file_menu():
    print("\nAvailable files to run:")
    for key, file in files.items():
        print(f"{key}. {file[0]}")
    print("0. Exit")

def run_file(file_tuple):
    """
    Rulează fișierul selectat utilizând funcția `main` asociată.
    """
    file_name, main_function = file_tuple
    print(f"Running {file_name}...")
    main_function()  # Apelează funcția principală din fișierul selectat

if __name__ == "__main__":
    while True:
        print_file_menu()
        choice = input("\nEnter the number of the file to run: ").strip()

        if choice == "0":
            print("Exiting...")
            break
        elif choice in files:
            run_file(files[choice])
        else:
            print("Invalid choice. Please try again.")
