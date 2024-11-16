import copy
import math
import os
from BreastCancerDataset import BreastCancerDataset
from model import MobileNetV3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.datasets.folder import default_loader


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

            # Backward propagation e otimizacao
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)  # Multiply by batch size

        epoch_loss = running_loss / len(train_loader.dataset)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {epoch_loss:.4f}\n")

    # Ensure the directory for the save_path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Salva o modelo ao final do treinamento
    torch.save(model, save_path)
    print(f"Model saved to {save_path}")


# Fluxo principal
def main():
    # Caminhos dos dados de treino e teste dataset/test
    train_data_path = "dataset/train"

    # Treino
    epochs = 1
    learning_rate = 0.0001  # Taxa de Aprendizado
    batch_size = 32  # Tamanho dos batches

    # Hyperparâmetros
    num_classes = 8  # Número de classes a serem previstas
    magnifications = ["40X", "100X", "200X", "400X"]  # Magnificacoes disponiveis
    criterion = nn.CrossEntropyLoss()  # Funcao de perda
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Usa cuda se existir gpu, caso contrario usa cpu

    # Inicializa o modelo small
    small_model = MobileNetV3(
        num_classes=num_classes,
        model_size="small",
        ks=True,
        ca=True,
        tr=False,
        sk=True,
    ).to(device)

    # Inicializa o modelo large
    large_model = MobileNetV3(
        num_classes=num_classes,
        model_size="large",
        ks=True,
        ca=True,
        tr=False,
        sk=True,
    ).to(device)

    # Otimizador
    optimizer_large = optim.Adam(large_model.parameters(), lr=learning_rate)
    optimizer_small = optim.Adam(small_model.parameters(), lr=learning_rate)

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
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # Normalização
            ),
        ]
    )

    # Carrega o dataset de treino com todas as magnificacoes
    train_dataset = BreastCancerDataset(
        root_dir=train_data_path, transform=train_transform
    )

    train_model(
        model=large_model,
        train_dataset=train_dataset,
        batch_size=batch_size,
        optimizer=optimizer_large,
        criterion=criterion,
        device=device,
        epochs=epochs,
        save_path="model/large_model.pth",
    )

    train_model(
        model=small_model,
        train_dataset=train_dataset,
        batch_size=batch_size,
        optimizer=optimizer_small,
        criterion=criterion,
        device=device,
        epochs=epochs,
        save_path="model/small_model.pth",
    )


if __name__ == "__main__":
    main()
