import cv2
from IPython.display import Image
import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import time
from scipy import ndimage as ndi
from scipy import signal
from skimage import graph, color, filters, morphology, segmentation, measure, feature

# Funciones para la adecuación de las imágenes
def loadImg(img): # Cargar imagen en RGB
    img = cv2.imread(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def imgYCrCb(img): # Espacio de color YCrCb
    return (cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))

def imgHSV(img): # Espacio de color HSV
    return (cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

def imgGRAY(img): # Espacio de color GRAY
    return (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

def blurImg(img, sizeMask):
    if sizeMask % 2 == 0:
        return 'La mascará debe ser impar'
    return cv2.GaussianBlur(img, (sizeMask,sizeMask), 0)

print('Optimo')