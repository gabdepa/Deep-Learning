import os
import shutil
import tarfile
from sklearn.model_selection import train_test_split

def clean_folder(paths):
    # Verifica se paths_to_delete é uma lista de strings
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

def extract_targz(file_path):
    # Extrai para o diretório especificado
    extract_path = "dataset/"

    # Abre e extrai
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(
            path=extract_path, filter="data"
        )  # 'data' preserva os dados sem alterações
    print(f"Arquivo {file_path} extraído em {extract_path}")

def organize_dataset(source_dir, train_dir, test_dir, test_size):
    categories = ["benign", "malignant"]
    magnifications = ["40X", "100X", "200X", "400X"]

    # Verifica se todas as entradas são strings
    if not all(
        isinstance(directory, str) for directory in [source_dir, train_dir, test_dir]
    ):
        raise TypeError("source_dir, train_dir, and test_dir must all be strings")

    for mag in magnifications:
        print(f"Processando magnificacao: {mag}")

        for category in categories:
            category_path = os.path.join(source_dir, category, "SOB")

            if not os.path.exists(category_path):
                print(f"Caminho não encontrado: {category_path}")
                continue  # Pular se a categoria não estiver presente

            for subtype in os.listdir(category_path):
                subtype_path = os.path.join(category_path, subtype)

                if not os.path.isdir(subtype_path):
                    continue  # Pula se não for diretorio

                for slide_id in os.listdir(subtype_path):
                    slide_id_path = os.path.join(subtype_path, slide_id)

                    if not os.path.isdir(slide_id_path):
                        continue  # Pula se não for diretorio

                    # Caminho pra pasta da magnificacao corrente
                    mag_path = os.path.join(slide_id_path, mag)
                    if not os.path.exists(mag_path):
                        continue  # Termina se diretorio não existir

                    # Coleta as imagens da pasta da magnificacao
                    all_images = [
                        os.path.join(mag_path, img)
                        for img in os.listdir(mag_path)
                        if img.endswith(".png")
                    ]

                    # Pula se não achar imagens
                    if len(all_images) == 0:
                        continue

                    # Verifica número de imagens antes de dividir
                    if len(all_images) >= 2:
                        # Divide entre teste e treino
                        train_images, test_images = train_test_split(
                            all_images, test_size=test_size, random_state=42
                        )
                    else:
                        # Apenas uma imagem, usa como treino
                        train_images = all_images.copy()

                    # Diretorios destino
                    train_dest = os.path.join(train_dir, mag, category, subtype)
                    test_dest = os.path.join(test_dir, mag, category, subtype)
                    os.makedirs(train_dest, exist_ok=True)
                    os.makedirs(test_dest, exist_ok=True)

                    # Copia imagens de treino
                    for img_path in train_images:
                        shutil.copy(img_path, train_dest)

                    # Copia imagens de teste
                    for img_path in test_images:
                        shutil.copy(img_path, test_dest)

                    print(
                        f"Completo para subtipo: {subtype}, slide {slide_id}, magnificacao {mag} com "
                        f"{len(train_images)} imagens de treino e {len(test_images)} imagens de teste."
                    )

    print("Organização do dataset concluída.")

# Caminhos iniciais e finais
breakhis_file = "dataset/BreaKHis_v1.tar.gz"
breakhis_dir = "dataset/BreaKHis_v1"
source_dir = "dataset/BreaKHis_v1/histology_slides/breast"
train_dir = "dataset/train"
test_dir = "dataset/test"

# Remover arquivos
clean_folder(paths=[test_dir, train_dir, breakhis_dir])

# Extrai tar.gz
extract_targz(breakhis_file)

# Executar a organização do dataset
organize_dataset(source_dir, train_dir, test_dir, test_size=0.2) # 80/20, conforme especificado no artigo
