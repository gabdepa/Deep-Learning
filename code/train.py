import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from BreastCancerDataset import BreastCancerDataset
from model import MobileNetV3
from datasetMeanValue import evaluateMeanStd
from filters import adaptive_histogram_equalization, enhance_colors, rgb_to_lab_transform

# Função de Treino
def train_model(model, train_dataset, batch_size, optimizer, criterion, device, epochs, model_name):
    """
    Esta função treina os modelos em um conjunto de dados utilizando o método de backpropagation e otimização.

    ### Parâmetros:
    - **model**: O modelo a ser treinado.
    - **train_dataset**: Dataset de treinamento contendo entradas e rótulos.
    - **batch_size**: Tamanho do batch usado no DataLoader.
    - **optimizer**: Otimizador para ajustar os pesos do modelo.
    - **criterion**: Função de perda para calcular o erro do modelo.
    - **device**: Dispositivo de execução (CPU ou GPU).
    - **epochs**: Número de épocas de treinamento.
    - **model_name**: Nome do arquivo para salvar o modelo treinado.

    ### Descrição:
    1. **Preparação dos Dados**:
    - Cria um `DataLoader` para carregar o conjunto de dados de treinamento em batches, garantindo que os dados sejam embaralhados para maior robustez durante o treinamento.

    2. **Treinamento**:
    - Para cada época:
        - Configura o modelo no modo de treinamento.
        - Inicializa a perda acumulada para a época.
        - Para cada batch de dados:
        - Move os dados para o dispositivo configurado (CPU ou GPU).
        - Realiza a passagem forward para calcular as predições do modelo.
        - Calcula a perda usando a função de perda especificada.
        - Realiza a retropropagação (backward pass) para ajustar os pesos do modelo.
        - Atualiza os pesos utilizando o otimizador.
        - Atualiza a perda acumulada.
        - Calcula e exibe a perda média por época.

    3. **Salvamento do Modelo**:
    - Garante que o diretório para salvar o modelo existe.
    - Salva o modelo treinado em um arquivo com o nome especificado.

    ### Resultados:
    - Durante o treinamento:
    - Imprime a perda média por época, fornecendo uma visão geral da evolução do treinamento.
    - Após o treinamento:
    - Salva o modelo treinado em um arquivo no diretório especificado.

    ### Aplicações:
    Essa função é usada para treinar os modelos, no dataset personalizado, possibilitando ajustes automáticos dos parâmetros do modelo.
    """
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

            running_loss += loss.item() * inputs.size(0)  # Multiplica pelo tamanho do batch

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Training Loss: {epoch_loss:.4f}\n")

    # Garante que o diretório para dir_path existe
    dir_path = "model/"
    os.makedirs(dir_path, exist_ok=True)
    
    model_name = f"{dir_path}{model_name}"
    # Salva o modelo ao final do treinamento
    torch.save(model, model_name)
    print(f"Modelo treinado salvo em: {model_name}\n")

# Fluxo principal
def main():
    # Caminhos dos dados de treino
    train_data_path = "dataset/train"

    # Hyperparâmetros
    epochs = 50 # Número de épocas, conforme artigo
    learning_rate = 0.0001  # Taxa de Aprendizado, conforme artigo
    batch_size = 32  # Tamanho dos batches
    num_classes = 8  # Número de classes a serem previstas
    criterion = nn.CrossEntropyLoss()  # Função de perda
    magnifications = ["40X", "100X", "200X", "400X"]  # Magnificações disponíveis
    
    # OBS: Preferível uso de GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa cuda se existir gpu, caso contrário usa cpu
    print(f"Usando: {device}")

    # Instâncialização do modelo "small"
    small_model = MobileNetV3(
        num_classes=num_classes,
        model_size="small",
        ks=True,
        ca=True,
        tr=False, # Não pode ser True, por conta da definição do modelo("small")
        sk=True
    ).to(device)

    # Instâncialização do modelo "large"
    large_model = MobileNetV3(
        num_classes=num_classes,
        model_size="large",
        ks=True,
        ca=True,
        tr=True,
        sk=True
    ).to(device)

    # Otimizador
    optimizer_large = optim.Adam(large_model.parameters(), lr=learning_rate) # Adam, utilizado no artigo
    optimizer_small = optim.Adam(small_model.parameters(), lr=learning_rate) # Adam, utilizado no artigo

    # Calculo de média e desvio padrão dos pixels das imagens de treino
    #mean, std = evaluateMeanStd(train_data_path)

    # Transformações conforme feito no artigo para treinar o modelo
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Redimensionar as imagens
        transforms.RandomHorizontalFlip(p=0.25), # Invertendo a imagem horizontalmente com probabilidade 25%
        transforms.RandomVerticalFlip(p=0.25), # Invertendo a imagem verticalmente com probabilidade 25%
        transforms.ToTensor(), # Convertendo imagens para tensores(Vetor de características)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalização
        #transforms.Normalize(mean=mean, std=std)
    ])

    # Carrega o dataset de treino com todas as magnificações
    train_dataset = BreastCancerDataset(root_dir=train_data_path, transform=train_transform)

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
        model_name="large_model.pth" 
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
        model_name="small_model.pth" 
    )

if __name__ == "__main__":
    main()