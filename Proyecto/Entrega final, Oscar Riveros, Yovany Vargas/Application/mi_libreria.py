# mi_libreria.py

import cv2
import numpy as np

def mi_funcion_de_procesamiento(image):
    # Ejemplo de procesamiento: convertir a escala de grises y detectar bordes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_colored
