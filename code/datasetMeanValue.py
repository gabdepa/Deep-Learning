from BreastCancerDataset import BreastCancerDataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def main():
    train_data_path = "dataset/train"

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Carregamento do seu conjunto de dados
    dataset = BreastCancerDataset(root_dir=train_data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Calculando a média e desvio padrão
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for images, _ in loader:
        for i in range(3):
            mean[i] += images[:, i, :, :].mean()
            std[i] += images[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))

    print(f"Mean: {mean}")
    print(f"Std: {std}")


if __name__ == "__main__":
    main()
