import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from BreastCancerDataset import BreastCancerDataset
from datasetMeanValue import evaluateMeanStd, rgbToLabTransform
from model import MobileNetV3


# Função de Treino
def train_model(
    model, train_dataset, batch_size, optimizer, criterion, device, epochs, save_path
):
    # Cria o DataLoader para treino
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward propagation e otimização
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(
                0
            )  # Multiplica pelo tamanho do batch

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {epoch_loss:.4f}\n")

    # Garante que o diretório para save_path existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Salva o modelo ao final do treinamento
    torch.save(model, save_path)
    print(f"Model saved to {save_path}")


# Fluxo principal
def main():
    # Caminhos dos dados de treino
    train_data_path = "dataset/train"

    # Treino
    epochs = 50
    learning_rate = 0.0001  # Taxa de Aprendizado
    batch_size = 32  # Tamanho dos batches

    # Hyperparâmetros
    num_classes = 8  # Número de classes a serem previstas
    magnifications = ["40X", "100X", "200X", "400X"]  # Magnificações disponíveis
    criterion = nn.CrossEntropyLoss()  # Função de perda
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Usa cuda se existir gpu, caso contrário usa cpu
    print(f"Usando: {device}")

    # Inicializa o modelo small
    small_model = MobileNetV3(
        num_classes=num_classes,
        model_size="small",
        ks=True,
        ca=True,
        tr=False,  # Não pode ser True, por conta da definição do modelo
        sk=True,
    ).to(device)

    # Inicializa o modelo large
    large_model = MobileNetV3(
        num_classes=num_classes, model_size="large", ks=True, ca=True, tr=True, sk=True
    ).to(device)

    # Otimizador
    optimizer_large = optim.Adam(
        large_model.parameters(), lr=learning_rate
    )  # Adam, utilizado no artigo
    optimizer_small = optim.Adam(
        small_model.parameters(), lr=learning_rate
    )  # Adam, utilizado no artigo

    # Calculo de média e desvio padrão dos pixels das imagens
    mean, std = evaluateMeanStd(train_data_path)

    # Transformações conforme feito no artigo
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Redimensionar as imagens
            transforms.RandomHorizontalFlip(
                p=0.25
            ),  # Invertendo a imagem horizontalmente com probabilidade 25%
            transforms.RandomVerticalFlip(
                p=0.25
            ),  # Invertendo a imagem verticalmente com probabilidade 25%
            transforms.ToTensor(),  # Convertendo imagens para tensores(Vetor de características)
            rgbToLabTransform,
            transforms.Normalize(mean=mean, std=std),  # Normalização
        ]
    )

    # Carrega o dataset de treino com todas as magnificações
    train_dataset = BreastCancerDataset(
        root_dir=train_data_path, transform=train_transform
    )

    # Treinando modelo "large" no conjunto de treino
    print(f'Treinando modelo "large".')
    train_model(
        model=large_model,
        train_dataset=train_dataset,
        batch_size=batch_size,
        optimizer=optimizer_large,
        criterion=criterion,
        device=device,
        epochs=epochs,
        save_path="model/large_model_with_filter.pth",
    )
    # Treinando modelo "small" no conjunto de treino
    print(f'Treinando modelo "small".')
    train_model(
        model=small_model,
        train_dataset=train_dataset,
        batch_size=batch_size,
        optimizer=optimizer_small,
        criterion=criterion,
        device=device,
        epochs=epochs,
        save_path="model/small_model_with_filter.pth",
    )


if __name__ == "__main__":
    main()
