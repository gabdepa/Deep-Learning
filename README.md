# MobileNetV3 Modificado para Classificação de Tipos de Câncer

Este repositório contém a implementação de uma versão modificada do MobileNetV3 para a classificação de tipos de câncer a partir de imagens histopatológicas. O código é baseado no artigo disponível em [MDPI](https://www.mdpi.com/2076-3417/14/17/7564).

## Referência do Código

A implementação do MobileNetV3 modificado pode ser encontrada no [GitHub](https://github.com/karryxz/Modified-model/blob/main/modified_mobilenetv3.py).

## Configuração do Ambiente

Para executar este projeto, você pode configurar um ambiente virtual para gerenciar as dependências. Use os seguintes comandos no bash:

```bash
# Verificar se o Python 3 está instalado
python3.10 --version

# Criar um ambiente virtual
python3.10 -m venv venv

# Ativar o ambiente virtual
source venv/bin/activate

# Instalar os pacotes necessários
pip install -r requirements.txt
```

## Conjunto de Dados

O conjunto de dados utilizado é o BreaKHis, descrito em detalhes no documento do [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/7312934/authors#authors). O conjunto de dados pode ser baixado pelo seguinte link:

[Baixar Conjunto de Dados BreaKHis](http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz)

### Configuração do Conjunto de Dados

Após baixar o arquivo `.tar.gz` pelo link fornecido, coloque-o na pasta `/dataset` dentro deste projeto. Este arquivo não será rastreado pelo Git devido ao seu grande tamanho. Para organizar o conjunto de dados em uma estrutura utilizável, execute o seguinte script:

```bash
python3 ./code/organizeDataset.py
```

Este script irá gerar os diretórios `train` e `test` sob `/dataset` com a seguinte configuração:

```
/dataset
├── train
│   ├── 100X
│   ├── 200X
│   ├── 400X
│   └── 40X
└── test
    ├── 100X
    ├── 200X
    ├── 400X
    └── 40X
```

Cada subdiretório contém duas categorias principais: `benigno` e `maligno`, que são subdivididos em subtipos como adenose, fibroadenoma, etc.

## Executando o Modelo

Uma vez que o conjunto de dados está preparado e o ambiente configurado, você pode treinar o modelo executando:

```bash
# Passo 1: Rodar ./code/organizeDataset.py
./code/organizeDataset.py

# Passo 2: Rodar ./code/train.py
# Isto irá treinar com o modelo Large primeiro, depois com o Small, tamanho referente ao modelo MobileNetV3
# Isso irá treinar e salvar os modelos na pasta ./model
./code/train.py

# Passo 3: Rodar o ./code/test.py
# Que irá entrar no diretório ./model, e para cada arquivo .pth rodar os testes
# Os resultados gerados serão salvos na pasta ./results
./code/test.py
```

## Formato dos Nomes de Arquivos de Imagens

As imagens no conjunto de dados são nomeadas de acordo com o seguinte formato:

`<PROCEDIMENTO_BIÓPSIA>_<CLASSE_TUMOR>_<TIPO_TUMOR>_<ANO>-<ID_LAMINA>-<AMPLIAÇÃO>-<SEQ>.png`

Onde:

- `PROCEDIMENTO_BIÓPSIA`: SOB (Biopsia Cirúrgica Aberta)
- `CLASSE_TUMOR`: B (Benigno) | M (Maligno)
- `TIPO_TUMOR`: A (Adenose) | F (Fibroadenoma) | PT (Tumor Filoide) | TA (Adenoma Tubular) ou DC (Carcinoma Ductal) | LC (Carcinoma Lobular) | MC (Carcinoma Mucinoso) | PC (Carcinoma Papilífero)
- `ANO`: DIGITO (ex.: 14 para 2014)
- `ID_LAMINA`: NÚMERO,SEÇÃO (ex.: 22549AB)
- `AMPLIAÇÃO`: 40|100|200|400
- `SEQ`: NÚMERO (Número de Sequência)

### Exemplo de Nome de Arquivo

```
SOB_B_A-14-22549AB-100-001.png
```

- `SOB`: Biopsia Cirúrgica Aberta
- `B`: Benigno
- `A`: Adenose
- `14`: Ano, 2014
- `22549AB`: Identificação da Lâmina
- `100`: Ampliação de 100x
- `001`: Número de Sequência
