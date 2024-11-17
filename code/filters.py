from PIL import ImageEnhance, Image
import numpy as np
import cv2


def enhance_colors(image):
    """Aumenta a saturação para realçar rosa e roxo."""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(
        1.5
    )  # Ajuste o fator conforme necessário para melhores resultados


def adaptive_histogram_equalization(image):
    """Aplica equalização de histograma adaptativa para melhorar o contraste."""
    np_image = np.array(image)
    # Convertendo para YUV para processar o canal Y
    yuv_img = cv2.cvtColor(np_image, cv2.COLOR_RGB2YUV)
    yuv_img[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
        yuv_img[:, :, 0]
    )
    output_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
    return Image.fromarray(output_img)


def rgb_to_lab_transform(image):
    """Converte imagem de RGB para Lab."""
    lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2Lab)
    return Image.fromarray(lab_image)
