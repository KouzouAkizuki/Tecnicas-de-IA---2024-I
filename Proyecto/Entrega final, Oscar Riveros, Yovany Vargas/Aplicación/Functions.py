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

# Funciones para la adecuación de las imágenes, hasta la obtención de bordes
def loadImg(path): # Cargar imagen en RGB
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def imgYCrCb(img): # Espacio de color YCrCb
    return (cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))

def blurImg(img): # Suavizado
    return cv2.GaussianBlur(img, (5,5), 0)

def imgGray(blur, color): # Escala de grises
    if color > 0: # YCrCb y HSV
        return blur[:,:,1]
    return cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

def imgBrightness(blur): # brillo
    yuc = cv2.cvtColor(blur, cv2.COLOR_RGB2YUV)
    y = yuc[:,:,0]
    return np.mean(y)

def otsuImg(blur, brightness, kernel): # Segmentación Otsu
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if brightness > 110:
        otsu = ~ otsu
    otsu = close(otsu, kernel)
    return otsu

def close(otsu, kernel): # Cierre
    return cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

def outEdgeImg(otsu, kernel): # Borde exterior
    dataEdge, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
    dataEdge = dataEdge[0] if len(dataEdge) == 2 else dataEdge[1]
    longestEdge = max(longestEdge, key = cv2.contourArea)
    edge = cv2.drawContours(np.zeros_like(otsu), [longestEdge], 0, (255, 255, 255), cv2.FILLED)
    return cv2.dilate(edge, kernel, iterations = 1)

def cannyImg(segmentedHand, kernel): # Canny
    canny = feature.Canny(segmentedHand, sigma = 1).astype(np.uint8)
    return cv2.dilate(canny, kernel, iterations = 1)

def shadowImg(segmentedHand, kernel):
    _,shadows = cv2.threshold(segmentedHand, 115, 255, cv2.THRESH_BINARY)
    return cv2.dilate(shadows, kernel, iterations = 1)

def AND(img1, img2, mask, kernel): # Operación AND (&&)
    firstOperation = cv2.bitwise_and(img1, img2, mask = mask)
    return cv2.dilate(firstOperation, kernel, iterations = 1)

def OR(img1, img2, mask, kernel): # Operación AND (&&)
    firstOperation = cv2.bitwise_or(img1, img2, mask = mask)
    return cv2.dilate(firstOperation, kernel, iterations = 1)

def getHandYCrCb(path):
    original = loadImg(path)
    blur = blurImg(imgYCrCb(original))
    brightness = imgBrightness(blur)
    kernel = np.ones((5, 5), np.uint8)
    otsu = otsuImg(blur, brightness, kernel)
    segmentedHand = AND(blur, blur, otsu)
    outEdge = outEdgeImg(otsu, kernel)
    canny = cannyImg(segmentedHand, kernel)
    shadows = shadowImg(segmentedHand, kernel)
    shadowCanny = AND(shadows, canny, None, kernel)
    hand = OR(outEdge, shadowCanny, None)
    return hand

img = getHandYCrCb('Images/Indoor_G.jpg')
cv2.imshow('Mano', img)
cv2.waitKey(0)
cv2.destroyAllWindows()