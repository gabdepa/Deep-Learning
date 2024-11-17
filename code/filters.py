import cv2
import numpy as np
from PIL import ImageEnhance, Image

def enhance_colors(image):
    """
    Esta função ajusta as cores de uma imagem, aumentando a saturação para realçar tonalidades como rosa e roxo.

    ### Parâmetros:
    - **image**: Uma imagem no formato `PIL.Image` a ser processada.

    ### Descrição:
    1. **Ajuste de Saturação**:
    - Utiliza o módulo `ImageEnhance.Color` do PIL para criar um objeto que permite alterar a saturação da imagem.
    - Aplica um fator de saturação de `1.5`, ajustado empiricamente para destacar cores específicas, como rosa e roxo.

    2. **Retorno**:
    - Retorna a imagem com a saturação aumentada, mantendo a mesma resolução e formato.

    ### Resultados:
    - Produz uma imagem com cores mais vibrantes, destacando tonalidades que podem ser relevantes para análises visuais ou modelos de aprendizado de máquina.

    ### Aplicações:
    Essa função é útil em processamento de imagens onde o realce de cores específicas pode melhorar a identificação de padrões visuais.
    """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1.5) # Ajuste de fator empiricamente

def adaptive_histogram_equalization(image):
    """
    Esta função aplica equalização de histograma adaptativa em uma imagem para melhorar o contraste.

    ### Parâmetros:
    - **image**: Uma imagem no formato `PIL.Image` a ser processada.

    ### Descrição:
    1. **Conversão para Formato Numérico**:
    - Converte a imagem de `PIL.Image` para um array NumPy para facilitar o processamento.

    2. **Transformação para Espaço de Cor YUV**:
    - Converte a imagem do espaço de cor RGB para YUV utilizando a função `cv2.cvtColor`.
    - Foca no canal de luminância (Y), que controla o brilho, enquanto mantém as informações de cor nos canais U e V.

    3. **Equalização de Histograma Adaptativa**:
    - Aplica a técnica de CLAHE (Contrast Limited Adaptive Histogram Equalization) no canal Y para melhorar o contraste de forma adaptativa, preservando detalhes em regiões claras ou escuras.
    - Define o limite de corte (`clipLimit`) e o tamanho da grade dos blocos (`tileGridSize`) para ajustar o nível de equalização.

    4. **Reconversão para RGB**:
    - Converte a imagem processada de volta para o espaço de cor RGB.

    5. **Retorno**:
    - Retorna a imagem processada como um objeto `PIL.Image`.

    ### Resultados:
    - Produz uma imagem com contraste aprimorado, destacando detalhes que podem ser perdidos em regiões escuras ou claras.

    ### Aplicações:
    Essa função é ideal para processar imagens, especialmente no cenário onde a melhoria do contraste facilita a detecção de padrões visuais ou características importantes.
    """
    np_image = np.array(image)
    # Convertendo para YUV para processar o canal Y
    yuv_img = cv2.cvtColor(np_image, cv2.COLOR_RGB2YUV)
    yuv_img[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(yuv_img[:, :, 0])
    output_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
    return Image.fromarray(output_img)

def rgb_to_lab_transform(image):
    """
    Esta função converte uma imagem do espaço de cores RGB para o espaço de cores Lab.

    ### Parâmetros:
    - **image**: Uma imagem no formato `PIL.Image` a ser convertida.

    ### Descrição:
    1. **Conversão para Formato NumPy**:
    - Converte a imagem de `PIL.Image` para um array NumPy para processamento.

    2. **Transformação de Espaço de Cor**:
    - Utiliza a função `cv2.cvtColor` do OpenCV para transformar a imagem do espaço de cor RGB para o espaço Lab.
    - O espaço de cor Lab representa cores em três componentes:
        - **L**: Luminosidade (brilho).
        - **a**: Componente de cor do verde ao vermelho.
        - **b**: Componente de cor do azul ao amarelo.

    3. **Reconversão para Formato de Imagem**:
    - Converte a imagem processada de volta para o formato `PIL.Image`.

    4. **Retorno**:
    - Retorna a imagem no espaço de cores Lab.

    ### Resultados:
    - Produz uma imagem no espaço de cores Lab, onde as informações de luminosidade e cor são separadas, facilitando certas operações de processamento de imagens.

    ### Aplicações:
    Essa função é útil em tarefas de visão computacional ou aprendizado de máquina que exigem manipulação ou análise separada de brilho e cor, como segmentação de imagem, detecção de bordas ou realce de características visuais.
    """
    lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2Lab)
    return Image.fromarray(lab_image)