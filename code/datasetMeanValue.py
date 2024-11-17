import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from BreastCancerDataset import BreastCancerDataset
from torchvision.transforms.functional import rgb_to_lab


def rgbToLabTransform(img):
    lab = rgb_to_lab(img)
    return lab


def evaluateMeanStd(data_path):
    """
    O código calcula a média e o desvio padrão dos pixels das imagens para normalizar o conjunto de dados.
    Proporcionando uma normalização mais precisa dos dados,
    • Precisão: Soma todos os valores de pixels e seus quadrados, considerando o total de pixels, resultando em cálculos estatísticos precisos.
    • Normalização Eficiente: Fornece médias e desvios padrão corretos para normalizar os dados, melhorando o desempenho do modelo.
    • Consistência: Evita distorções causadas por tamanhos de lote ou dimensões de imagem diferentes.
    Args:
        data_path (String): Caminho para diretório
    """
    # Resize de 224x224
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

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
