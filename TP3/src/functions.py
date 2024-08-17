
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow
import imutils
from imutils.object_detection import non_max_suppression


def template_matching(input_image, template_image, min_scale=0.1, max_scale=3, num_scales=120, visualize=False):
    # Cargar la imagen de plantilla y convertirla a escala de grises y a bordes
    template = cv.imread(template_image)
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    template = cv.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]

    # Cargar la imagen de entrada y convertirla a escala de grises
    image = cv.imread(input_image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    found = None

    # Recorrer las escalas de la imagen
    for scale in np.linspace(min_scale, max_scale, num_scales)[::-1]:
        # Redimensionar la imagen según la escala y calcular la relación de redimensionamiento
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # Si la imagen redimensionada es más pequeña que la plantilla, salir del bucle
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # Detectar bordes en la imagen redimensionada y en escala de grises, y aplicar coincidencia de plantillas
        edged = cv.Canny(resized, 50, 200)
        result = cv.matchTemplate(edged, template, cv.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        # Comprobar si se debe visualizar la iteración
        if visualize:
            # Dibujar un recuadro alrededor de la región detectada
            clone = np.dstack([edged, edged, edged])
            cv.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)


        # Actualizar la variable de seguimiento si se ha encontrado un nuevo valor de correlación máximo
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # Desempaquetar la variable de seguimiento y calcular las coordenadas (x, y) del recuadro de detección
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # Dibujar un recuadro alrededor del resultado detectado y mostrar la imagen
    if visualize:
        cv.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

    return image

def read_from_path(path):
    """

    """
    # Lista para guardar lo encontrado
    images = []
    for filename in os.listdir(path):
        # Verifica la extensión que requiere
        if filename.endswith(".png") or filename.endswith(".jpg"):
            images.append(os.path.join(path, filename))
    return images
