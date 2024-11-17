import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from BreastCancerDataset import BreastCancerDataset
from datasetMeanValue import evaluateMeanStd
from filters import adaptive_histogram_equalization, enhance_colors, rgb_to_lab_transform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Função de teste
def test_model(model, data_path, magnifications, batch_size, device, model_file, results_dir, zero_division_value=1):
    """
    Esta função realiza a avaliação do modelo para diferentes níveis de magnificação de imagens. Ela calcula métricas de desempenho como acurácia, precisão, recall e F1-Score, além de gerar matrizes de confusão e gráficos de desempenho.

    ### Parâmetros:
    - **model**: O modelo treinado a ser avaliado.
    - **data_path**: Caminho para o diretório contendo os dados de teste.
    - **magnifications**: Lista de níveis de magnificação para os quais o modelo será avaliado.
    - **batch_size**: Tamanho do batch usado no DataLoader.
    - **device**: Dispositivo de execução (CPU ou GPU).
    - **model_file**: Nome base do arquivo do modelo, usado para salvar resultados.
    - **results_dir**: Diretório onde os resultados serão salvos.
    - **zero_division_value**: Valor padrão usado para evitar erros de divisão por zero nas métricas.

    ### Descrição:
    1. **Preparação dos Dados**:
    - Define as transformações para pré-processamento das imagens, incluindo redimensionamento, normalização e/ou realce de cores, equalização adaptativa e conversão para o formato Lab.

    2. **Avaliação do Modelo**:
    - Para cada nível de magnificação:
        - Carrega o dataset correspondente e cria um DataLoader.
        - Executa o modelo no conjunto de teste, coletando predições e rótulos verdadeiros.
        - Calcula métricas de desempenho (accuracy, precision, recall, F1-Score).
        - Gera e salva a matriz de confusão correspondente.

    3. **Métricas Gerais**:
    - Calcula métricas agregadas considerando todas as magnificações.
    - Gera a matriz de confusão geral e a salva.

    4. **Visualização de Métricas**:
    - Organiza os resultados em um DataFrame e cria um gráfico comparativo das métricas para diferentes magnificações.
    - Salva o gráfico em um arquivo.

    ### Resultados:
    - Matrizes de confusão para cada nível de magnificação e para o conjunto geral.
    - Métricas de desempenho detalhadas para cada magnificação e gerais.
    - Gráfico comparativo das métricas.

    Esta função é utilizada para avaliar o desempenho de modelos treinados em dados com diferentes características visuais, como imagens com diferentes níveis de ampliação.
    """    
    # Inicializa as listas de resultados gerais
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    magnification_levels = []
    overall_predictions = []
    overall_labels = []

    # Calculo de média e desvio padrão dos pixels das imagens
    mean, std = evaluateMeanStd(data_path)

    # Transformações para avaliação do modelo treinado
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Redimensionar as imagens
        transforms.ToTensor(),  # Convertendo imagens para tensores(Vetor de características)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalização
    ])

    for magnification in magnifications:
        print(f"Testing for magnification level: {magnification}")
        
        # Carrega o dataset para a magnificação atual
        test_dataset_magnification = BreastCancerDataset(
            root_dir=data_path,
            magnifications=[magnification],
            transform=test_transform
        )

        test_loader = DataLoader(test_dataset_magnification, batch_size=batch_size, shuffle=False)

        # Avalia o modelo para a magnificação atual
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

        # Agrega predições e rótulos nas métricas gerais
        overall_predictions.extend(all_predictions)
        overall_labels.extend(all_labels)

        # Calcula métricas para magnificações atuais
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=zero_division_value
        )
        recall = recall_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=zero_division_value
        )
        f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=zero_division_value
        )
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # Adiciona métricas as listas
        magnification_levels.append(magnification)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Plota e salva matriz de confusão
        cm_filename = f"{model_file}_confusion_matrix_{magnification}.png"
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=test_dataset_magnification.class_names,
            yticklabels=test_dataset_magnification.class_names
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title(f"Confusion Matrix - {magnification}")
        plt.savefig(os.path.join(results_dir, cm_filename), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Métricas para magnificação {magnification}:")
        print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}\n")

    # Calcula métricas totais para todas as magnificações
    overall_accuracy = accuracy_score(overall_labels, overall_predictions)
    overall_precision = precision_score(
        overall_labels,
        overall_predictions,
        average="macro",
        zero_division=zero_division_value
    )
    overall_recall = recall_score(
        overall_labels,
        overall_predictions,
        average="macro",
        zero_division=zero_division_value
    )
    overall_f1 = f1_score(
        overall_labels,
        overall_predictions,
        average="macro",
        zero_division=zero_division_value
    )
    overall_conf_matrix = confusion_matrix(overall_labels, overall_predictions)

    # Adiciona métricas as listas
    magnification_levels.append("Overall")
    accuracies.append(overall_accuracy)
    precisions.append(overall_precision)
    recalls.append(overall_recall)
    f1_scores.append(overall_f1)

    # Plota e salva a matriz geral
    overall_cm_filename = f"{model_file}_confusion_matrix_overall.png"
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        overall_conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=test_dataset_magnification.class_names,
        yticklabels=test_dataset_magnification.class_names
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix - Overall")
    plt.savefig(os.path.join(results_dir, overall_cm_filename), dpi=300, bbox_inches="tight")
    plt.close()

    # Imprime métricas
    print("Métricas Gerais:")
    print(f"Accuracy: {overall_accuracy:.3f}, Precision: {overall_precision:.3f}, Recall: {overall_recall:.3f}, F1-Score: {overall_f1:.3f}\n")
    print("Matriz de Confusão Geral salva como:", overall_cm_filename)

    # Métricas para cada magnificação e geral
    metrics_df = pd.DataFrame({
            "Magnification": magnification_levels,
            "Accuracy": accuracies,
            "Precision": precisions,
            "Recall": recalls,
            "F1-Score": f1_scores
    })

    metrics_melted = metrics_df.melt(
        id_vars="Magnification",
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Value"
    )

    magnification_order = magnifications + ["Overall"]
    metrics_melted["Magnification"] = pd.Categorical(
        metrics_melted["Magnification"], 
        categories=magnification_order,
        ordered=True
    )

    # Plota as métricas
    metrics_graph_filename = f"{model_file}_metrics.png"
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Magnification", y="Value", hue="Metric", data=metrics_melted)
    plt.ylim(0, 1)
    plt.title("Model Performance Metrics")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, metrics_graph_filename), dpi=300, bbox_inches="tight")
    plt.close()
    print("Gráfico de Métricas salvo como:", metrics_graph_filename)

# Fluxo principal
def main():
    # Diretório de Treino
    test_data_path = "dataset/test"
    
    # Diretório para métricas de resultados
    results_dir = "results"
    # Garante que o diretório para resultados existe
    os.makedirs(results_dir, exist_ok=True)

    # Diretório para os modelos
    models_path = "model"
    
    # Magnificações disponíveis
    magnifications = ["40X", "100X", "200X", "400X"] 
    # Tamanho dos batches
    batch_size = 32  

    # OBS: Preferível uso de GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa cuda se existir gpu, caso contrário usa cpu
    print(f"Usando: {device}")

    for model_file in os.listdir(models_path):
        # Verifica se arquivo é do formato PyTorch
        if model_file.endswith(".pth"):
            print(f"Modelo: {model_file}")
            model_path = os.path.join(models_path, model_file)
            # Carrega o modelo completo
            model = torch.load(model_path, weights_only=False).to(device)
            # Testando modelo da iteração atual no conjunto de teste
            test_model(
                model=model,
                data_path=test_data_path,
                magnifications=magnifications,
                batch_size=batch_size,
                device=device,
                model_file=model_file,
                results_dir=results_dir,
                zero_division_value=1
            )

if __name__ == "__main__":
    main()