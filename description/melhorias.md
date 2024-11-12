# Melhorias Potenciais para Modelos de Rede Neural

## Estratégias de Melhoria e Justificativa

### 1. Normalização por Lote

- **Descrição**: Melhora a estabilidade da rede neural durante o treinamento, normalizando as entradas de cada mini-lote.
- **Benefícios**: Acelera o treinamento e melhora o desempenho do modelo final, reduzindo o problema do deslocamento interno da covariância.

### 2. Aumento de Dados

- **Descrição**: Utiliza transformações como rotações, inversões e variações de cor para aumentar a robustez do modelo contra variações nos dados de entrada.
- **Contexto Específico**: Em imagens médicas, simular variações reais nos procedimentos de captura de imagem pode ajudar o modelo a lidar melhor com variações naturais nos dados.

### 3. Dropout

- **Descrição**: Técnica de regularização aplicada após camadas de ativação ou blocos de atenção para evitar overfitting.
- **Justificativas**:
  - **Normalização por Lote e Dropout**: A normalização por lote ajuda na eficiência do treinamento e o dropout reduz o risco de overfitting, essencial em redes complexas como o MobileNetV3.
  - **Aumento de Dados e Dropout**: O aumento de dados promove a robustez e o dropout assegura que o modelo não se torne excessivamente dependente de características específicas, promovendo uma generalização mais ampla.
  - **Atenção Mecânica e Dropout**: Blocos de atenção focam em características cruciais e o dropout após esses blocos previne que o modelo se torne excessivamente especializado nessas características, favorecendo uma aprendizagem mais distribuída.

## Implementação do Dropout nos Blocos Bneck

### Localização do Dropout

- **Após Funções de Ativação**: O Dropout é idealmente colocado após camadas de ativação dentro dos blocos bneck. Isso permite que ele funcione sobre as ativações não-lineares, reduzindo a dependência do modelo em neurônios específicos, o que promove uma aprendizagem mais robusta e generalizada.
- **Antes de Operações de Fusão e Agregação**: Aplicar Dropout antes de blocos que sintetizam ou agregam informações pode ser estratégico para assegurar que a rede mantenha a capacidade de generalizar bem, sem sobreajustar aos detalhes específicos dos dados de treinamento.

### Interação com Mecanismos de Atenção

- **Blocos de Atenção como SE e CA**: Ao aplicar Dropout após blocos de atenção, ajudamos a garantir que o modelo não se torne excessivamente confiante nas características enfatizadas por estes mecanismos. Isso é particularmente importante pois esses blocos ajustam a importância dos canais com base nas características globais, que podem variar significativamente entre diferentes conjuntos de dados ou mesmo entre diferentes casos dentro do mesmo conjunto.

### Interação com Funções de Ativação

- **Funções de Ativação H-swish e ReLU**: Estas funções são usadas nos blocos bneck para introduzir não-linearidades. Aplicar Dropout após essas funções de ativação ajuda a promover uma representação de características que é robusta a pequenas variações e ruídos nos dados de entrada, essencial para modelos que operam em ambientes computacionais limitados ou que processam dados médicos complexos.
