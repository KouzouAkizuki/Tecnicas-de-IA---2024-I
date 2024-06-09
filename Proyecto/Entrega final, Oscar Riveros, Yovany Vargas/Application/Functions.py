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
def loadImg(route): # Cargar imagen en RGB
    route = os.path.dirname(__file__) + route
    img = cv2.imread(route, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def imgYCrCb(img): # Espacio de color YCrCb
    return (cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))

def imgHSV(img): # Espacio de color HSV
    return (cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

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

def otsuImg(imgGray, brightness): # Segmentación Otsu
    _, otsu = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if brightness > 110:
        otsu = ~ otsu
    return otsu

def outEdgeImg(otsu, area, kernel, n): # Borde exterior
    dataEdge = cv2.findContours(otsu, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
    dataEdge = dataEdge[0] if len(dataEdge) == 2 else dataEdge[1]
    longestEdge = max(dataEdge, key = cv2.contourArea)
    method = cv2.FILLED if area else 3
    edge = cv2.drawContours(np.zeros_like(otsu), [longestEdge], 0, (255, 255, 255), method)
    return cv2.dilate(edge, kernel, iterations = n)

def cannyImg(segmentedHand, kernel, n): # Canny
    canny = feature.canny(segmentedHand, sigma=1).astype(np.uint8)*255
    return cv2.dilate(canny, kernel, iterations = n)

def shadowImg(segmentedHand, kernel, n): # Sombras
    _,shadows = cv2.threshold(segmentedHand, 115, 255, cv2.THRESH_BINARY)
    return cv2.dilate(shadows, kernel, iterations = n)

def AND(img1, img2, otsu, kernel, n): # Operación AND (&&)
    firstOperation = cv2.bitwise_and(img1, img2, mask = otsu)
    return cv2.dilate(firstOperation, kernel, iterations = n)

def OR(img1, img2, mask, kernel, n): # Operación AND (&&)
    firstOperation = cv2.bitwise_or(img1, img2, mask = mask)
    return cv2.dilate(firstOperation, kernel, iterations = n)

def close(img, kernel, n): # Cierre
    output = img
    for _ in range(n):
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    return output

def getHand(path):
    original = loadImg(path)
    blurYCrCb = blurImg(imgYCrCb(original))
    blurHSV = blurImg(imgHSV(original))
    brightness = imgBrightness(blurHSV)
    kernel = np.ones((7, 7), np.uint8)
    grayYCrCb = imgGray(blurYCrCb, 1) # Para bordes externos, silueta
    grayHSV = imgGray(blurHSV, 1) # Para bordes internos, dedos
    otsu = otsuImg(grayYCrCb, brightness)
    areaHand = outEdgeImg(otsu, True, kernel, 0)
    segmentedHand = AND(grayHSV, grayHSV, areaHand, kernel, 0)
    canny = cannyImg(segmentedHand, kernel, 1)
    shadows = shadowImg(segmentedHand, kernel, 1)
    shadowCanny = AND(shadows, canny, None, kernel, 0)
    outEdge = outEdgeImg(otsu, False, kernel, 0)
    hand = OR(outEdge, shadowCanny, None, kernel, 0)
    initial = np.zeros_like(hand)
    while not np.array_equal(initial, hand):
        initial = hand
        hand = close(hand, kernel, 1)
    return hand