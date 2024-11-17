import os
import shutil
import tarfile
from sklearn.model_selection import train_test_split

def clean_folder(paths):
    """
    Esta função remove arquivos e diretórios especificados em uma lista de caminhos.

    ### Parâmetros:
    - **paths**: Lista de strings contendo os caminhos absolutos ou relativos dos arquivos e diretórios que devem ser removidos.

    ### Descrição:
    1. **Validação dos Parâmetros**:
    - Verifica se o argumento `paths` é uma lista de strings.
    - Caso contrário, lança um erro do tipo `TypeError`.

    2. **Remoção de Arquivos e Diretórios**:
    - Para cada caminho na lista:
        - Se for um diretório, utiliza `shutil.rmtree` para remover o diretório e todo o seu conteúdo.
        - Se for um arquivo, utiliza `os.remove` para removê-lo.
        - Caso o caminho não exista, exibe uma mensagem informando que o caminho não foi encontrado.

    3. **Mensagens de Status**:
    - Após a remoção de cada arquivo ou diretório, exibe mensagens no console indicando a ação realizada.

    ### Resultados:
    - Remove arquivos e diretórios especificados na lista `paths`.
    - Emite mensagens informativas para cada ação realizada ou para caminhos não encontrados.

    ### Aplicações:
    Essa função é útil para limpar diretórios ou remover arquivos específicos, garantindo que apenas os arquivos e diretórios desejados sejam mantidos.
    """

    # Verifica se paths_to_delete é uma lista de strings
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise TypeError("paths must be a list of strings")

    # Função para deletar arquivos e diretórios
    for path in paths:
        if os.path.isdir(path):
            shutil.rmtree(path) # Remove diretórios e seus conteúdos
            print(f"Diretório removido: {path}")
        elif os.path.isfile(path):
            os.remove(path) # Remove arquivos
            print(f"Arquivo removido: {path}")
        else:
            print(f"Caminho não encontrado para remoção: {path}")

def extract_targz(file_path):
    """
    Esta função extrai o conteúdo do arquivo `BreakHis.tar.gz` para o diretório especificado "dataset/".

    ### Parâmetros:
    - **file_path**: Caminho para o arquivo `.tar.gz` a ser extraído.

    ### Descrição:
    1. **Definição do Diretório de Extração**:
    - Define o diretório de destino padrão como `dataset/`, onde o conteúdo do arquivo será extraído.

    2. **Extração do Arquivo**:
    - Abre o arquivo `.tar.gz` usando o módulo `tarfile` no modo de leitura comprimido (`r:gz`).
    - Extrai todo o conteúdo para o diretório `dataset/` utilizando `extractall`.

    3. **Mensagens de Status**:
    - Após a extração, exibe uma mensagem no console indicando que o arquivo foi extraído com sucesso e o diretório de destino.

    ### Resultados:
    - O conteúdo do arquivo `.tar.gz` é extraído para o diretório `dataset/`.

    ### Aplicações:
    Essa função é útil para automatizar o processo de extração do dataset comprimido.
    """

    # Extrai para o diretório "dataset"
    extract_path = "dataset/"

    # Abre e extrai
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_path, filter="data")  # 'data' preserva os dados sem alterações
    print(f"Arquivo {file_path} extraído em {extract_path}")

def organize_dataset(source_dir, train_dir, test_dir, test_size):
    """
   Esta função organiza o conjunto de dados de imagens, dividindo-o em conjuntos de treino e teste com base em categorias, subtipos e níveis de magnificação.

    ### Parâmetros:
    - **source_dir**: Caminho para o diretório contendo as imagens brutas organizadas por categorias e subtipos.
    - **train_dir**: Caminho para o diretório onde as imagens de treino serão salvas.
    - **test_dir**: Caminho para o diretório onde as imagens de teste serão salvas.
    - **test_size**: Proporção do conjunto de dados a ser usada para teste (entre 0 e 1).

    ### Descrição:
    1. **Definição de Categorias e Magnificações**:
    - Define as categorias de classificação ("benign" e "malignant") e os níveis de magnificação das imagens ("40X", "100X", "200X", "400X").

    2. **Validação dos Diretórios**:
    - Verifica se os parâmetros de diretórios são strings válidas, lançando um erro se não forem.

    3. **Organização do Dataset**:
    - Para cada nível de magnificação e categoria:
        - Navega pelos subdiretórios contendo subtipos e slide IDs.
        - Verifica a existência do diretório correspondente à magnificação.
        - Coleta todas as imagens no formato `.png`.
        - Divide as imagens entre treino e teste utilizando `train_test_split` com base no tamanho do conjunto de teste.
        - Copia as imagens para os diretórios de destino (`train_dir` e `test_dir`), criando-os se necessário.

    4. **Mensagens de Status**:
    - Exibe mensagens no console indicando o progresso para cada subtipo, slide e nível de magnificação, incluindo o número de imagens em cada conjunto.

    5. **Conclusão**:
    - Após processar todas as categorias e níveis de magnificação, exibe uma mensagem indicando que a organização foi concluída.

    ### Resultados:
    - Diretórios de treino e teste contendo imagens organizadas por categorias, subtipos e níveis de magnificação.
    - Mensagens detalhadas sobre o progresso e possíveis problemas, como diretórios ou imagens ausentes.

    ### Aplicações:
    Essa função é útil para organizar dados brutos em conjuntos de treino e teste, garantindo uma estrutura hierárquica clara e consistente para diferentes categorias e níveis de magnificação.
    """

    categories = ["benign", "malignant"]
    magnifications = ["40X", "100X", "200X", "400X"]

    # Verifica se todas as entradas são strings
    if not all(isinstance(directory, str) for directory in [source_dir, train_dir, test_dir]):
        raise TypeError("source_dir, train_dir, and test_dir must all be strings")

    for mag in magnifications:
        print(f"Processando magnificação: {mag}")
        for category in categories:
            category_path = os.path.join(source_dir, category, "SOB")

            if not os.path.exists(category_path):
                print(f"Caminho não encontrado: {category_path}")
                continue  # Pular se a categoria não estiver presente

            for subtype in os.listdir(category_path):
                subtype_path = os.path.join(category_path, subtype)

                if not os.path.isdir(subtype_path):
                    continue  # Pula se não for diretório

                for slide_id in os.listdir(subtype_path):
                    slide_id_path = os.path.join(subtype_path, slide_id)

                    if not os.path.isdir(slide_id_path):
                        continue  # Pula se não for diretório

                    # Caminho pra pasta da magnificação corrente
                    mag_path = os.path.join(slide_id_path, mag)
                    if not os.path.exists(mag_path):
                        continue  # Termina se diretório não existir

                    # Coleta as imagens da pasta da magnificação
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
                        # Divide entre treino e teste
                        train_images, test_images = train_test_split(all_images, test_size=test_size, random_state=42)
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

                    print(f"Completo para subtipo: {subtype}, slide {slide_id}, magnificação {mag} com {len(train_images)} imagens de treino e {len(test_images)} imagens de teste.")

    print("Organização do dataset concluída.")

# Caminhos iniciais e finais
breakhis_file = "dataset/BreaKHis_v1.tar.gz"
breakhis_dir = "dataset/BreaKHis_v1"
source_dir = "dataset/BreaKHis_v1/histology_slides/breast"
train_dir = "dataset/train"
test_dir = "dataset/test"
pycache_dir = "code/__pycache__"

# Remover arquivos
clean_folder(paths=[test_dir, train_dir, breakhis_dir, pycache_dir])

# Extrai tar.gz
extract_targz(breakhis_file)

# Executar a organização do dataset
organize_dataset(source_dir, train_dir, test_dir, test_size=0.2) # 80/20, conforme especificado no artigo
