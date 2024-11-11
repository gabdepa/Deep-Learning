import os
import shutil
import tarfile
import subprocess
from sklearn.model_selection import train_test_split

# Caminhos dos arquivos e diretórios que desejamos excluir
paths_to_delete = [
    'dataset/BreaKHis_v1',
    'dataset/BreaKHis_v1.tar.gz',
    'dataset/test',
    'dataset/train'
]

# Função para deletar arquivos e diretórios
for path in paths_to_delete:
    if os.path.isdir(path):
        shutil.rmtree(path)  # Remove diretórios e seus conteúdos
        print(f"Diretório removido: {path}")
    elif os.path.isfile(path):
        os.remove(path)  # Remove arquivos
        print(f"Arquivo removido: {path}")
    else:
        print(f"Caminho não encontrado para remoção: {path}")

# Command to concatenate the parts into a single tar.gz file
command = 'cat dataset/BreaKHis_v1_part_* > dataset/BreaKHis_v1.tar.gz'
# Execute the command
subprocess.run(command, shell=True, check=True)
print("Concatenated parts into BreaKHis_v1.tar.gz")

# Path to the tar.gz file
file_path = 'dataset/BreaKHis_v1.tar.gz'
# Extract to the specified directory
extract_path = 'dataset/'  # Modify as needed

# Open and extract
with tarfile.open(file_path, 'r:gz') as tar:
    tar.extractall(path=extract_path, filter='data')  # 'data' preserva os dados sem alterações
print(f"Extracted {file_path} to {extract_path}")

# Caminhos iniciais e finais
source_dir = "dataset/BreaKHis_v1/histology_slides/breast"
train_dir = "dataset/train"
test_dir = "dataset/test"

# Função para organizar os dados
def organize_dataset():
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
            
            # Dividir em treino e teste (80% - 20%)
            train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)
            
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

# Executar a organização do dataset
organize_dataset()
print("Organização do dataset concluída.")
