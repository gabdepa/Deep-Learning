# Plano de Treinamento do Modelo de Classificação de Subtipos de Câncer de Mama

Este documento descreve detalhadamente o plano de treinamento, incluindo a divisão de dados, técnicas de otimização e parâmetros de treinamento empregados na classificação de subtipos em imagens de patologia de câncer de mama utilizando uma rede neural profunda.

## Divisão dos Dados

Para o treinamento do modelo, os dados foram divididos seguindo a proporção de 80% para o conjunto de treinamento e 20% para o conjunto de validação. Essa divisão permite uma avaliação adequada da generalização do modelo em dados não vistos, mantendo uma quantidade substancial de dados para o treinamento efetivo.

## Técnicas de Treinamento

### Otimizador Adam

O modelo utiliza o otimizador Adam, conhecido por sua eficiência em cenários de treinamento de redes profundas. Adam ajusta as taxas de aprendizado de forma adaptativa para diferentes parâmetros a partir de estimativas de primeiros e segundos momentos dos gradientes, facilitando a convergência rápida e estável em espaços de alta dimensionalidade.

### Taxa de Aprendizado e Tamanho do Lote

A taxa de aprendizado inicial foi definida como 0.0001, com um tamanho de lote de 32. Esses parâmetros foram escolhidos para equilibrar a velocidade de treinamento e a estabilidade do processo de aprendizado. O tamanho de lote moderado permite uma estimativa suficientemente precisa do gradiente, enquanto mantém a eficiência computacional.

### Funcionalidade de Perda

O treinamento foi guiado pela função de perda de entropia cruzada, que mede a discrepância entre as distribuições de probabilidade previstas pelo modelo e as verdadeiras distribuições de probabilidade dos rótulos. A entropia cruzada é adequada para tarefas de classificação, pois penaliza predições incorretas de forma mais significativa, acelerando o aprendizado em casos de alta confiança errônea.

### Estratégias de Regularização

Para mitigar o overfitting, técnicas de regularização como o dropout e a normalização por lote foram aplicadas durante o treinamento. O dropout desativa aleatoriamente uma proporção de neurônios durante o treinamento, enquanto a normalização por lote ajusta e escala as ativações para melhorar a estabilidade e desempenho do modelo.

### Aumento de Dados

O conjunto de treinamento foi submetido a técnicas de aumento de dados para melhorar a robustez do modelo contra variações nos dados de entrada. As técnicas incluíram rotações, zoom e inversões horizontais, que ajudam o modelo a generalizar melhor para novas imagens, simulando diferentes condições de captura.

## Configuração do Ambiente de Treinamento

O treinamento foi realizado em uma máquina equipada com uma GPU NVIDIA GeForce RTX 3080 Ti, utilizando o framework Pytorch. O ambiente de desenvolvimento incluiu suporte a CUDA para aceleração por GPU, garantindo treinamento eficiente e rápido.

## Critérios de Avaliação

Os critérios de avaliação foram baseados em métricas padrão de classificação, incluindo precisão, recall, valor F1 e acurácia. Cada métrica foi calculada usando médias macro para garantir uma avaliação justa e equitativa entre as categorias.
