import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from BreastCancerDataset import BreastCancerDataset

def evaluateMeanStd(data_path):
    r"""
    Esta função calcula a média e o desvio padrão dos pixels das imagens em um conjunto de dados, facilitando a normalização durante o pré-processamento.

    ### Parâmetros:
    - **data_path**: Caminho (string) para o diretório contendo o conjunto de dados de imagens.

    ### Descrição:
    1. **Definição de Transformações**:
    - Define uma sequência de transformações:
        - Redimensionamento das imagens para 224x224 pixels.
        - Conversão das imagens para tensores, permitindo cálculos numéricos.

    2. **Carregamento do Dataset**:
    - Utiliza a classe `BreastCancerDataset` para carregar o conjunto de dados com as transformações definidas.
    - Cria um `DataLoader` para facilitar a iteração pelas imagens em batches.

    3. **Cálculo de Estatísticas**:
    - Inicializa tensores de zeros para armazenar a soma e soma dos quadrados dos valores dos pixels em cada canal (R, G, B).
    - Itera por todas as imagens no DataLoader:
        - Calcula a soma dos valores de pixels e a soma dos quadrados.
        - Atualiza o número total de pixels processados.
    - Calcula a média dividindo a soma dos valores pelo número total de pixels.
    - Calcula o desvio padrão usando a fórmula: \\(\sqrt{\text{soma dos quadrados / total de pixels} - \text{média}^2}\).

    4. **Exibição dos Resultados**:
    - Exibe os valores calculados de média e desvio padrão para cada canal (R, G, B).

    5. **Retorno**:
    - Retorna dois tensores: `mean` (média) e `std` (desvio padrão), ambos com três valores, correspondendo aos canais R, G e B.

    ### Resultados:
    - Valores de média e desvio padrão precisos para cada canal de cor do conjunto de dados, que podem ser usados para normalizar as imagens durante o treinamento.

    ### Aplicações:
    Essa função é essencial para o pré-processamento de imagens, garantindo que os dados sejam normalizados de forma consistente.
    """
    # Transformações para cálculo de média e desvio padrão
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize para 224x224
        transforms.ToTensor() # Convertendo imagens para tensores(Vetor de características)
    ])

    # Carregamento do conjunto de dados
    dataset = BreastCancerDataset(root_dir=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Calculando a média e desvio padrão
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0
    for images, _ in loader:
        batch_pixels = images.size(0) * images.size(2) * images.size(3)
        total_pixels += batch_pixels
        mean += images.sum(dim=[0, 2, 3])
        std += (images**2).sum(dim=[0, 2, 3])
    mean /= total_pixels
    std = torch.sqrt(std / total_pixels - mean**2)

    print(f"Média para {data_path}: {mean}")
    print(f"Desvio Padrão {data_path}: {std}")
    return mean, std