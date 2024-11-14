import copy
import math
from code.model import MobileNetV3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Função de Treino
def train_model(
    model,
    train_loader,
    optimizer,
    criterion=nn.CrossEntropyLoss(),
    device=torch.device("cuda"),
):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# Função de Treino vista em Aula
def train_model_2(
    model,
    optimizer,
    criterion=nn.CrossEntropyLoss(),
    max_epochs=10,
    grace_period=3,
    device=torch.device("cuda"),
):
    best_loss = math.inf
    curr_grace_period = 0
    best_model = copy.deepcopy(model.state_dict())

    for epoch in range(max_epochs):
        print(f"Época {epoch+1}/{max_epochs}")
        print("-" * 10)

        for phase in ["treino", "validacao"]:
            if phase == "treino":
                model.train()  # Colocar o modelo em modo de treino
            else:
                model.eval()  # Colocar o modelo em modo de avaliação

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "treino"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "treino":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            print()

            if phase == "validacao":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    curr_grace_period = 0
                    best_model = copy.deepcopy(model.state_dict())
                else:
                    curr_grace_period += 1
                    if curr_grace_period >= grace_period:
                        print("Early stopping")
                        model.load_state_dict(best_model)
                        return

    model.load_state_dict(best_model)
    return


# Função de Teste
def test_model(model, test_loader, zero_devision_value=0, device=torch.device("cuda")):
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

    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_predictions)
    # zero_division=1: Define a precisão como 1.0 para classes sem previsões positivas, evitando warnings em classes raras.
    precision = precision_score(
        all_labels, all_predictions, average="macro", zero_division=zero_devision_value
    )
    recall = recall_score(
        all_labels, all_predictions, average="macro", zero_division=zero_devision_value
    )
    f1 = f1_score(
        all_labels, all_predictions, average="macro", zero_division=zero_devision_value
    )

    return accuracy, precision, recall, f1


# Fluxo principal
def main():
    # Hyperparâmetros
    learning_rate = 0.0001  # Taxa de Aprendizado
    momentum_value = 0.9  # Valor de momentum
    batch = 32  # Tamanho dos batchs

    # Caminhos dos dados de treino e teste dataset/test
    train_data_path = "dataset/train"
    test_data_path = "dataset/test"

    # Transformações conforme feito no artigo
    article_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Redimensionar as imagens
            transforms.ToTensor(),  # Convertendo imagens para tensores(Vetor de características)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalização
            transforms.RandomHorizontalFlip(
                p=0.25
            ),  # Invertendo a imagem da esquerda para a direita com probabilidade p 25%
            transforms.RandomVerticalFlip(
                p=0.25
            ),  # Invertendo a imagem de cima para baixo com probabilidade p 25%
        ]
    )

    # Carregar o dataset
    train_dataset = datasets.ImageFolder(
        root=train_data_path, transform=article_transform
    )
    test_dataset = datasets.ImageFolder(
        root=test_data_path, transform=article_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # Número de classes a serem previstas
    num_classes = 8
    # Configurações do Modelo
    model = MobileNetV3(
        num_classes=num_classes, model_size="small", ks=True, ca=True, tr=False, sk=True
    )
    # OBRIGATÓRIO: Utilizar GPU("cuda")
    model = model.to(torch.device("cuda"))

    # Função de Perda
    criterion = nn.CrossEntropyLoss()

    # Otimizador
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_value)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loop Treinamento e Avaliação no conjunto de Teste
    epochs = 50  # Número de épocas máximo
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion)
        accuracy, precision, recall, f1 = test_model(
            model, test_loader, zero_devision_value=1
        )

        print(f"Época [{epoch+1}/{epochs}]")
        print(f"Loss de Treinamento: {train_loss:.4f}")
        print(f"Métricas calculadas no conjunto de Teste:")
        print(
            f"Acurácia: {accuracy:.3f}, Precisão: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}\n"
        )

    # OPCIONAL: Salvar o modelo treinado
    # torch.save(model.state_dict(), "mobilenetv3_breast_cancer.pth")


if __name__ == "__main__":
    main()
