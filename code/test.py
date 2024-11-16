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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torchvision.datasets.folder import default_loader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Redimensionar as imagens
        transforms.ToTensor(),  # Convertendo imagens para tensores(Vetor de características)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # Normalização
        ),
    ]
)


def test_model(
    model,
    test_data_path,
    magnifications,
    batch_size,
    device,
    model_file,
    results_dir,
    zero_division_value=1,
):
    # I# Inicializa as listas de resultados gerais
    magnification_levels = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    overall_predictions = []
    overall_labels = []

    for magnification in magnifications:
        # Carrega o dataset para a mag atual
        test_dataset_magnification = BreastCancerDataset(
            root_dir=test_data_path,
            magnifications=[magnification],
            transform=test_transform,
        )

        print(f"Testing for magnification level: {magnification}")

        test_loader = DataLoader(
            test_dataset_magnification, batch_size=batch_size, shuffle=False
        )

        # Avalia o modelo para a magnificacao atual
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

        # Agrega predicoes e rotulos nas metricas gerais
        overall_predictions.extend(all_predictions)
        overall_labels.extend(all_labels)

        # Calcula metricas para magnificacoes atuais
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=zero_division_value,
        )
        recall = recall_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=zero_division_value,
        )
        f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=zero_division_value,
        )
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # Adiciona metricas as listas
        magnification_levels.append(magnification)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Plota e salva matriz de confusao
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=test_dataset_magnification.class_names,
            yticklabels=test_dataset_magnification.class_names,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title(f"Confusion Matrix - {magnification}")
        cm_filename = f"{model_file}_confusion_matrix_{magnification}.png"
        plt.savefig(
            os.path.join(results_dir, cm_filename), dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"Metrics for magnification {magnification}:")
        print(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
            f"Recall: {recall:.3f}, F1-Score: {f1:.3f}\n"
        )

    # Calcula metricas totais para todas as magnificacoes
    overall_accuracy = accuracy_score(overall_labels, overall_predictions)
    overall_precision = precision_score(
        overall_labels,
        overall_predictions,
        average="macro",
        zero_division=zero_division_value,
    )
    overall_recall = recall_score(
        overall_labels,
        overall_predictions,
        average="macro",
        zero_division=zero_division_value,
    )
    overall_f1 = f1_score(
        overall_labels,
        overall_predictions,
        average="macro",
        zero_division=zero_division_value,
    )
    overall_conf_matrix = confusion_matrix(overall_labels, overall_predictions)

    # Adiciona metricas as listas
    magnification_levels.append("Overall")
    accuracies.append(overall_accuracy)
    precisions.append(overall_precision)
    recalls.append(overall_recall)
    f1_scores.append(overall_f1)

    # Plota e salva a matriz geral
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        overall_conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=test_dataset_magnification.class_names,
        yticklabels=test_dataset_magnification.class_names,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix - Overall")
    overall_cm_filename = f"{model_file}_confusion_matrix_overall.png"
    plt.savefig(
        os.path.join(results_dir, overall_cm_filename), dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("Overall metrics:")
    print(
        f"Accuracy: {overall_accuracy:.3f}, Precision: {overall_precision:.3f}, "
        f"Recall: {overall_recall:.3f}, F1-Score: {overall_f1:.3f}\n"
    )
    print("Overall confusion matrix saved as:", overall_cm_filename)

    # Metricas para cada mag e geral
    metrics_df = pd.DataFrame(
        {
            "Magnification": magnification_levels,
            "Accuracy": accuracies,
            "Precision": precisions,
            "Recall": recalls,
            "F1-Score": f1_scores,
        }
    )

    metrics_melted = metrics_df.melt(
        id_vars="Magnification",
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Value",
    )

    magnification_order = magnifications + ["Overall"]
    metrics_melted["Magnification"] = pd.Categorical(
        metrics_melted["Magnification"], categories=magnification_order, ordered=True
    )

    # Plota as metricas
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Magnification", y="Value", hue="Metric", data=metrics_melted)
    plt.ylim(0, 1)
    plt.title("Model Performance Metrics")
    plt.legend(loc="lower right")
    metrics_graph_filename = f"{model_file}_metrics.png"
    plt.savefig(
        os.path.join(results_dir, metrics_graph_filename), dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("Metrics graph saved as:", metrics_graph_filename)


def main():
    print("Current working directory:", os.getcwd())
    test_data_path = "dataset/test"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    models_path = "model"
    magnifications = ["40X", "100X", "200X", "400X"]  # Magnificacoes disponiveis
    batch_size = 32  # Tamanho dos batches
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Usa cuda se existir gpu, caso contrario usa cpu

    for model_file in os.listdir(models_path):
        if model_file.endswith(".pth"):  # Verifica se arquivo eh do formato PyTorch
            print(f"{model_file}")
            model_path = os.path.join(models_path, model_file)
            # Carrega o modelo completo
            model = torch.load(model_path).to(device)

            test_model(
                model=model,
                test_data_path=test_data_path,
                magnifications=magnifications,
                batch_size=batch_size,
                device=device,
                model_file=model_file,
                results_dir=results_dir,
                zero_division_value=1,
            )


if __name__ == "__main__":
    main()
