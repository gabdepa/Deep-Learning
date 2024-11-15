import copy
import math
import os
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


def train_model(model, train_loader, optimizer, criterion, device):
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
    return epoch_loss


def test_model(model, test_loader, zero_division_value=0, device=None):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels, all_predictions, average="macro", zero_division=zero_division_value
    )
    recall = recall_score(
        all_labels, all_predictions, average="macro", zero_division=zero_division_value
    )
    f1 = f1_score(
        all_labels, all_predictions, average="macro", zero_division=zero_division_value
    )

    return accuracy, precision, recall, f1


# Custom Dataset class to handle nested directory structure
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, magnifications=None, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_names = [
            "adenosis",
            "fibroadenoma",
            "phyllodes_tumor",
            "tubular_adenoma",
            "ductal_carcinoma",
            "lobular_carcinoma",
            "mucinous_carcinoma",
            "papillary_carcinoma",
        ]
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }
        if magnifications is None:
            magnifications = os.listdir(root_dir)
        for magnification in magnifications:
            magnification_dir = os.path.join(root_dir, magnification)
            if os.path.isdir(magnification_dir):
                for diagnosis in ["benign", "malignant"]:
                    diagnosis_dir = os.path.join(magnification_dir, diagnosis)
                    if os.path.isdir(diagnosis_dir):
                        for subtype in os.listdir(diagnosis_dir):
                            subtype_dir = os.path.join(diagnosis_dir, subtype)
                            if os.path.isdir(subtype_dir):
                                label = self.class_to_idx[subtype]
                                for img_name in os.listdir(subtype_dir):
                                    img_path = os.path.join(subtype_dir, img_name)
                                    self.image_paths.append(img_path)
                                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = default_loader(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Fluxo principal
def main():
    # Hyperparâmetros
    learning_rate = 0.0001  # Taxa de Aprendizado
    batch_size = 32  # Tamanho dos batchs
    num_classes = 8  # Número de classes a serem previstas
    magnifications = ["40X", "100X", "200X", "400X"]  # Magnificacoes disponiveis
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Usa cuda se existir gpu, caso contrario usa cpu

    # Caminhos dos dados de treino e teste dataset/test
    train_data_path = "dataset/train"
    test_data_path = "dataset/test"

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

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Redimensionar as imagens
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

    # Cria o DataLoader para treino
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Inicializa o modelo
    model = MobileNetV3(
        num_classes=num_classes,
        model_size="small",
        ks=True,
        ca=True,
        tr=False,
        sk=True,
    ).to(device)

    # Funcao de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Treino
    epochs = 1
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {train_loss:.4f}\n")

    # Testa o modelo para cada nivel de magnificacao
    for mag in magnifications:
        print(f"Testing for magnification level: {mag}")

        # Carrega o dataset de teste pra a magnificacao
        test_dataset = BreastCancerDataset(
            root_dir=test_data_path, magnifications=[mag], transform=test_transform
        )

        # Garante que as classes sao consistentes entre os datasets de treino e teste
        assert (
            train_dataset.class_to_idx == test_dataset.class_to_idx
        ), "Classes nao correspondentes entre os datasets de treino e teste."

        # Cria o DataLoader de teste
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Avalia o modelo para a magnificacao atual
        accuracy, precision, recall, f1 = test_model(
            model, test_loader, zero_division_value=1, device=device
        )

        print(f"Metricas de teste para a magnificacao {mag}:")
        print(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}\n"
        )


if __name__ == "__main__":
    main()
