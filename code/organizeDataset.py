import os
import shutil
import tarfile
import subprocess
from sklearn.model_selection import train_test_split

def clean_folder(paths):
    # Check if paths_to_delete is a list of strings
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise TypeError("paths must be a list of strings")
    
    # Função para deletar arquivos e diretórios
    for path in paths:
        if os.path.isdir(path):
            shutil.rmtree(path)  # Remove diretórios e seus conteúdos
            print(f"Diretório removido: {path}")
        elif os.path.isfile(path):
            os.remove(path)  # Remove arquivos
            print(f"Arquivo removido: {path}")
        else:
            print(f"Caminho não encontrado para remoção: {path}")

def restore_targz():
    # Command to concatenate the parts into a single tar.gz file
    command = 'cat dataset/BreaKHis_v1_part_* > dataset/BreaKHis_v1.tar.gz'
    # Execute the command
    subprocess.run(command, shell=True, check=True)
    print("Concatenando partes de BreaKHis_v1_part_* em BreaKHis_v1.tar.gz")

def extract_targz():
# Path to the tar.gz file
    file_path = 'dataset/BreaKHis_v1.tar.gz'
    # Extract to the specified directory
    extract_path = 'dataset/'  # Modify as needed

    # Open and extract
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_path, filter='data')  # 'data' preserva os dados sem alterações
    print(f"Arquivo {file_path} extraído em {extract_path}")

def organize_dataset(source_dir, train_dir, test_dir, size):
    # Check if all inputs are strings
    if not all(isinstance(directory, str) for directory in [source_dir, train_dir, test_dir]):
        raise TypeError("source_dir, train_dir, and test_dir must all be strings")
    
    categories = ['benign', 'malignant']
    for category in categories:
        category_path = os.path.join(source_dir, category, "SOB")
        
        if not os.path.exists(category_path):
            print(f"Caminho não encontrado: {category_path}")
            continue  # Pular se a categoria não estiver presente
        
        for subtype in os.listdir(category_path):
            subtype_path = os.path.join(category_path, subtype)
            all_images = []

            # Coletar todas as imagens nas subpastas de forma recursiva
            for root, _, files in os.walk(subtype_path):
                images = [os.path.join(root, img) for img in files if img.endswith('.png')]
                all_images.extend(images)

            # Verificação se há imagens para dividir
            if len(all_images) == 0:
                print(f"Nenhuma imagem encontrada para o subtipo: {subtype}")
                continue
            
            # Dividir em treino e teste
            train_images, test_images = train_test_split(all_images, test_size=size, random_state=42)
            
            # Mover imagens para as pastas de treino e teste
            for img_path in train_images:
                dest = os.path.join(train_dir, subtype)
                os.makedirs(dest, exist_ok=True)
                shutil.copy(img_path, dest)
            
            for img_path in test_images:
                dest = os.path.join(test_dir, subtype)
                os.makedirs(dest, exist_ok=True)
                shutil.copy(img_path, dest)
            
            print(f"Organização concluída para o subtipo {subtype} com {len(train_images)} imagens de treino e {len(test_images)} de teste.")
    print("Organização do dataset concluída.")

# Caminhos iniciais e finais
breakhis_file = "dataset/BreaKHis_v1.tar.gz"
breakhis_dir = "dataset/BreaKHis_v1"
source_dir = "dataset/BreaKHis_v1/histology_slides/breast"
train_dir = "dataset/train"
test_dir = "dataset/test"

# Remover arquivos
clean_folder(paths=[breakhis_dir, breakhis_file, test_dir, train_dir])
# Restaurar arquivo tar.gz original
restore_targz()
# Extrai tar.gz
extract_targz()
# Executar a organização do dataset
organize_dataset(source_dir, train_dir, test_dir, size=0.3)
# clean_folder(paths=[breakhis_dir, breakhis_file])