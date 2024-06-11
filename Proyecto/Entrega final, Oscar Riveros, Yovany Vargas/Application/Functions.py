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
    return cv2.GaussianBlur(img, (3,3), 0)

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
    return ~otsu if brightness < 130 and brightness > 110 else otsu

def outEdgeImg(otsu, area): # Borde exterior
    dataEdge = cv2.findContours(otsu, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
    dataEdge = dataEdge[0] if len(dataEdge) == 2 else dataEdge[1]
    longestEdge = max(dataEdge, key = cv2.contourArea)
    method = cv2.FILLED if area else 3
    edge = cv2.drawContours(np.zeros_like(otsu), [longestEdge], 0, (255, 255, 255), method)
    return edge

def cannyImg(segmentedHand, kernel, n): # Canny
    canny = feature.canny(segmentedHand, sigma=1).astype(np.uint8)*255
    return cv2.dilate(canny, kernel, iterations = n)

def shadowImg(segmentedHand, kernel, n): # Sombras
    _,shadows = cv2.threshold(segmentedHand, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.dilate(shadows, kernel, iterations = n)

def AND(img1, img2, kernel, n): # Operación AND (&&)
    operation = cv2.bitwise_and(img1, img2)
    return cv2.dilate(operation, kernel, iterations = n)

def OR(img1, img2, kernel, n): # Operación AND (&&)
    firstOperation = cv2.bitwise_or(img1, img2)
    return cv2.dilate(firstOperation, kernel, iterations = n)

def close(img, kernel, n): # Cierre
    output = img
    for _ in range(n):
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    return output

def open(img, kernel, n):
    output = img
    for _ in range(n):
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
    return output

def lessBins(img, bins):
    normalized_img = cv2.normalize(img, None, 0, bins-1, cv2.NORM_MINMAX)
    return(normalized_img)

def superClose(img, kernel,n):
    dilated = cv2.erode(img, kernel, iterations = n)
    dilaroded = cv2.dilate(dilated, kernel, iterations = n)
    return dilaroded

def getHand(path):
    original = loadImg(path) # Imagen original el RGB
    brightness = imgBrightness(original)
    print(brightness)
    kernel = np.ones((5, 5), np.uint8)
    blurYCrCb = blurImg(imgYCrCb(original)) # Suavizado de la imagen en YCrCb
    YCrCb16 = lessBins(blurYCrCb[:,:,1], 16)
    otsu = superClose(otsuImg(YCrCb16, brightness), kernel, 20)
    handArea = outEdgeImg(otsu, True)
    outEdge = outEdgeImg(otsu, False)
    segmentedHand = AND(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), handArea, None, 0)
    canny = cannyImg(segmentedHand, kernel, 1)
    shadows = shadowImg(segmentedHand, kernel, 0)
    shadowCanny = AND(shadows,canny, kernel, 4)
    hand = OR(outEdge, shadowCanny, kernel, 0)
    initial = np.zeros_like(hand)
    while not np.array_equal(initial, hand):
        initial = hand
        hand = close(hand, kernel, 1)
    return hand

# Funciones para la extracción de características
def handLines(areaHand, hand):
    return cv2.bitwise_and(areaHand, cv2.bitwise_not(hand))

def distanceTransform(areaHand):
    return cv2.distanceTransform(areaHand, distanceType=cv2.DIST_L1, maskSize=3)