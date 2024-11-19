# Descrição Detalhada dos Componentes do Modelo de Rede Neural

## 1. Classes de Módulos e Blocos

### SKModel

- **Descrição**: Define um bloco que usa mecanismos de atenção baseados em kernels seletivos (SK). Ajusta dinamicamente os tamanhos dos kernels de convolução com base em informações aprendidas, permitindo que o modelo se concentre em características importantes.
- **Funcionalidade**: O SKModel utiliza múltiplos kernels de tamanhos diferentes para processar a entrada e combinar os resultados, ajustando a atenção aos detalhes mais e menos granulares conforme necessário.

### Funções de Ativação: h_sigmoid e h_swish

- **h_swish**: Variação da função swish, otimizada para eficiência computacional em ambientes de hardware limitado.
- **h_sigmoid**: Versão modificada da função sigmóide, usada para calcular a ativação h_swish.

### CoordAtt

- **Descrição**: Bloco de atenção coordenada que melhora o foco do modelo em localizações específicas dentro da imagem.
- **Funcionalidade**: Divide a atenção ao longo das dimensões horizontal e vertical, permitindo que o modelo capture informações espaciais cruciais de maneira mais eficaz.

### SaELayer

- **Descrição**: Camada de atenção que utiliza pooling adaptativo e duas vias de conexões densas.
- **Funcionalidade**: Recalibra canais de características com base em informações espaciais reduzidas, melhorando a sensibilidade do modelo a características importantes independentemente da sua posição na imagem.

### conv_block_mo

- **Descrição**: Função utilitária para criar um bloco de convolução padronizado com normalização e ativação.
- **Funcionalidade**: Usada em várias partes da rede para construir blocos convolucionais, facilitando a modularidade e a reutilização do código.

### SEblock

- **Descrição**: Bloco de excitação e espremimento (SE) que recalibra os canais de características.
- **Funcionalidade**: Aumenta significativamente a performance do modelo ao modelar interdependências entre canais, permitindo um foco aprimorado nas características relevantes.

### HireAtt

- **Descrição**: Arquitetura de atenção hierárquica.
- **Funcionalidade**: Combina informações de múltiplas camadas de características para formar uma representação mais rica, usada nas previsões finais.

### bneck

- **Descrição**: Define um bloco de "bottleneck" com várias opções para incorporar mecanismos de atenção.
- **Funcionalidade**: Componente chave na arquitetura MobileNetV3, que permite uma aprendizagem eficiente e eficaz de características representativas.

## 2. Classe Principal MobileNetV3

- **Descrição**: Estrutura a arquitetura geral do modelo, definindo a sequência de blocos bneck e outras operações de convolução.
- **Variabilidade**: Diferencia-se em 'small' ou 'large' com variações na capacidade e profundidade, adaptando-se a diferentes requisitos computacionais e de performance.

## 3. Scripts de Treinamento e Teste

### train.py

- **Descrição**: Executa um ciclo de treinamento do modelo, gerenciando o processo de otimização.
- **Funcionalidade**: Inclui o cálculo da perda e a atualização dos pesos do modelo via backpropagation.

### test.py

- **Descrição**: Avalia o desempenho do modelo em um conjunto de teste.
- **Funcionalidade**: Calcula métricas como acurácia, precisão, recall e F1-score, essenciais para avaliar a aplicabilidade do modelo.
