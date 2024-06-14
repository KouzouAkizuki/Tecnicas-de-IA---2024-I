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

def imgGray(img): # Escala de grises
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def imgBrightness(blur): # brillo
    yuc = cv2.cvtColor(blur, cv2.COLOR_RGB2YUV)
    y = yuc[:,:,0]
    return np.mean(y)

def otsuImg(imgGray, brightness): # Segmentación Otsu
    _, otsu = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ~otsu if brightness < 130 and brightness > 110 else otsu

def outEdgeImg(otsu): # Borde exterior
    dataEdge = cv2.findContours(otsu, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
    dataEdge = dataEdge[0] if len(dataEdge) == 2 else dataEdge[1]
    longestEdge = max(dataEdge, key = cv2.contourArea)
    area = cv2.drawContours(np.zeros_like(otsu), [longestEdge], 0, (255, 255, 255), cv2.FILLED)
    edge = cv2.drawContours(np.zeros_like(otsu), [longestEdge], 0, (255, 255, 255), 3)
    return area, edge

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
    return cv2.normalize(img, None, 0, bins-1, cv2.NORM_MINMAX)

def superClose(img, kernel,n):
    dilated = cv2.erode(img, kernel, iterations = n)
    dilaroded = cv2.dilate(dilated, kernel, iterations = n)
    return dilaroded

def getHand(path):
    original = loadImg(path) # Imagen original RGB
    y, x = original.shape[:2]
    brightness = imgBrightness(original) # Cálculo del brillo
    print('Brillo:', brightness) # Mostrar el brillo en consola, para posibles errores más adelante
    kernel = np.ones((5, 5), np.uint8) # Se declara un kernel de 5x5 inferior a esto, no es capaz de eliminar el ruido
    blurYCrCb = blurImg(imgYCrCb(original)) # Suavizado de histograma
    YCrCb16 = lessBins(blurYCrCb[:,:,1], 16) # Reducción de bins
    otsu = superClose(otsuImg(YCrCb16, brightness), kernel, 20) # Se aplica un close
    handArea, outEdge = outEdgeImg(otsu) # Se obtiene el area y borde de la mano
    segmentedHand = AND(imgGray(original), handArea, None, 0) # Mano completa en GRAY
    canny = cannyImg(segmentedHand, kernel, 2) # Bordes Canny
    shadows = shadowImg(segmentedHand, kernel, 2) # Sombras
    shadowCanny = AND(shadows,canny, kernel, 2) # And entre sombras y canny
    hand = OR(outEdge, shadowCanny, kernel, 0) # Or entre borde exterior y and anterior
    initial = np.zeros_like(hand)
    while not np.array_equal(initial, hand): # Se hacen cierres hasta que no se detecten más cambios
        initial = hand
        hand = close(hand, kernel, 1)
    rect = cv2.rectangle(np.zeros_like(blurYCrCb[:,:,0]),(0,0),(x,y),(255,255,255),50)
    handArea = cv2.bitwise_and(cv2.bitwise_not(rect), handArea)
    hand, handArea, x, y = reSizeHand(hand, handArea, x, y)
    return hand, handArea, x, y, original

def reSizeHand(hand, handArea, x, y):
    x = x//10
    y = y//10
    handArea = cv2.resize(handArea, (x, y), fx = 0, fy = 0, interpolation = cv2.INTER_LINEAR)
    handArea[handArea > 0] = 255
    hand = cv2.resize(hand, (x, y), fx = 0, fy = 0, interpolation = cv2.INTER_LINEAR)
    hand[hand > 0] = 255
    case = np.any(hand[0, :] == 255)
    case = (case and True) if (np.any(hand[:, -1] == 255) or np.any(hand[:, 0] == 255)) else False
    if case: # Esta rotación es más por comodidad que por necesidad
        hand, handArea = cv2.rotate(hand, cv2.ROTATE_180), cv2.rotate(handArea, cv2.ROTATE_180)
    return hand, handArea, x, y

# Funciones para la extracción de características
def handContour(hand, handArea):
    return cv2.bitwise_and(cv2.bitwise_not(hand), handArea)

def distanceTransform(img, x):
    outPut = cv2.distanceTransform(img, distanceType = cv2.DIST_L1, maskSize = 3)
    maxDistance = np.max(outPut)
    indexMaxPoint = np.argmax(outPut)
    coordinateMaxPoint = (indexMaxPoint % x, indexMaxPoint // x)
    return outPut, maxDistance, coordinateMaxPoint

def getWristPoints(handArea, maxDistance, palmCenter, x, y):
    angle = np.linspace(0,360,1000)
    circle = np.zeros_like(angle)
    radio = int(maxDistance * 1.5)
    indexX = np.clip((radio * np.sin(angle * np.pi/180) + palmCenter[0]).astype(int), 0, x - 1) # x
    indexY = np.clip((radio * np.cos(angle * np.pi/180) + palmCenter[1]).astype(int), 0, y - 1) # y
    for i in range(len(circle)): # Extraer valores del area
        circle[i] = handArea[indexY[i]][indexX[i]]
    shift = np.argmin(circle)
    circle = np.roll(circle, -shift) # Desplazar el valor mínimo al inicio
    circle[-1] = 0 # Ultimo cero
    circleDistance, maxCircleDistance, _ = distanceTransform(circle.astype(np.uint8), x)
    maxCircleDistance = maxCircleDistance.astype(int)
    coordinateMaxPoint = np.argmax(circleDistance).astype(int)
    circle = np.roll(circle, maxCircleDistance + len(angle) - coordinateMaxPoint)
    indexY = np.roll(indexY, maxCircleDistance + len(angle) - coordinateMaxPoint - shift)
    indexX = np.roll(indexX, maxCircleDistance + len(angle) - coordinateMaxPoint - shift)
    circleDistance = np.roll(circleDistance, maxCircleDistance + len(angle) - coordinateMaxPoint)
    circleDistance = circleDistance.flatten()
    distancePeak = signal.find_peaks(circleDistance, height=10, width=5)
    distancePeak = distancePeak[0].astype(int)
    if(len(distancePeak) > 6):
            distancePeak = distancePeak[:6]
    radiusPeak = circleDistance[distancePeak].astype(int)
    leftWrist = np.vstack((indexX[distancePeak - radiusPeak], indexY[distancePeak - radiusPeak])).T[0]
    rightWrist = np.vstack((indexX[distancePeak + radiusPeak], indexY[distancePeak + radiusPeak])).T[0]
    centerWrist = (leftWrist + rightWrist)/2
    return centerWrist, leftWrist, maxDistance

def findLinePoints(initialPoint, m, b, radio):
    x1, y1 = initialPoint
    coefA = 1 + m**2
    coefB = 2*m*b - 2*x1 - 2*y1*m
    coefC = -(radio**2 - x1**2 - y1**2 + 2*y1*b - b**2)
    newX = np.roots([coefA, coefB, coefC])
    newY = m*newX + b
    return newX, newY

def findPointsOrientation(palmCenter, centerWrist, xTried, yTried):
    x0, y0 = palmCenter
    x1, y1 = centerWrist
    x1Larger = x1 > x0
    x1Equal = x1 == x0
    y1Larger = y1 > y0
    y1Equal = x1 == x0
    if y1Equal: # EN eje Y pero no en X
        index = np.argmin(yTried) if x1Larger else np.argmax(yTried)
    elif x1Equal: # En eje X pero no en Y
        index = np.argmax(xTried) if y1Larger else np.argmin(xTried)
    elif x1Larger:
        index = np.argmax(xTried) if y1Larger else np.argmin(yTried)
    elif not x1Larger:
        index = np.argmax(yTried) if y1Larger else np.argmin(xTried)
    left =  (xTried[index], yTried[index])
    right = (xTried[~index], yTried[~index])
    return left, right

def thumbLinePoints(palmCenter, centerWrist, leftWrist, radio, factor):
    radio = radio*factor
    palmCenter = [palmCenter[0], -palmCenter[1]]
    centerWrist = [centerWrist[0], -centerWrist[1]]
    leftWrist = [leftWrist[0], -leftWrist[1]]
    x1, y1, = centerWrist
    x2, y2 = leftWrist
    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    newX, newY = findLinePoints(centerWrist, m, b, radio)
    initialLeft, initialRight = findPointsOrientation(palmCenter, centerWrist, newX, newY)
    newM = -1/m
    bLeft = initialLeft[1] - newM * initialLeft[0]
    leftX, leftY = findLinePoints(initialLeft, newM, bLeft, radio*factor)
    finalLeft, _ = findPointsOrientation(centerWrist, initialLeft, leftX, leftY)
    bRight = initialRight[1] - newM * initialRight[0]
    rightX, rightY = findLinePoints(initialRight, newM, bRight, radio*factor)
    _, finalRight = findPointsOrientation(centerWrist, initialRight, rightX, rightY)
    initialLeft = np.abs(np.array(initialLeft).astype(np.int32))
    finalLeft = np.abs(np.array(finalLeft).astype(np.int32))
    initialRight = np.abs(np.array(initialRight).astype(np.int32))
    finalRight = np.abs(np.array(finalRight).astype(np.int32))
    return initialLeft, finalLeft, initialRight, finalRight, m, b

def restFingers(initialLeft, initialRight, centerPalm, centerWrist):
    
    return None