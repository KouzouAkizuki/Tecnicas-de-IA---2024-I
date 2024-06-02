# Librerias
import cv2
from skimage import feature, filters
from scipy import signal
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import time
import numpy as np
import csv
import os


featuresVector = ['Numero de dedos', 'Angulo indice', 'Angulo corazon', 'Angulo anular', 'Angulo menique',
                  'Ancho pulgar', 'Ancho indice', 'Ancho corazon', 'Ancho anular', 'Ancho menique',
                  'Pulgar extendido', 'Indice extendido', 'Corazon extendido', 'Anular extendido', 'Menique extendido',
                  'Pulgar contraido', 'Indice contraido', 'Corazon contraido', 'Anular contraido', 'Menique contraido',
                  'Hu_1', 'Hu_2', 'Hu_3', 'Hu_4', 'Hu_5', 'Hu_6', 'Hu_7', 'Redondez', 'Compacidad','Clase']

# Funciones
def HSV(img): # Cambio de espacio de color
    return (cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

def blurImg(img): # Suavizado del histograma y escala de grises
    #Se suaviza el histograma para eliminar minimos locales
    blur = cv2.GaussianBlur(HSV(img), (3,3), 0)
    return cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

def otsu(imgBlur): #Calculo del umbral por otsu
    #Calculo del brillo de la imagen
    y,x = imgBlur.shape[:2]
    brillo = np.sum(imgBlur)/(x*y)
    #La mano serA un mAximo global, para garantizar su aparicion se usa umbral binario
    tre, otsu = cv2.threshold(imgBlur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Inversion del umbral en dado caso que el brillo de la imagen sea mayor a 110
    if(brillo>110):
            otsu = ~otsu

    #Dilatar el umbral Otsu para rellenar Huecos
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(otsu, kernel, iterations = 2)

def edge(otsu): # Bordes
    #Hallar los contornos de la imagen
    c_Var = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_Var = c_Var[0] if len(c_Var) == 2 else c_Var[1]
    return max(c_Var, key = cv2.contourArea)
        
def segmentation(imgBlur, imgEdge): # Segmentacion
    #Obtener el contorno y su version rellena sin borde
    Contorno_Fill   = np.zeros_like(imgBlur)
    Contorno        = np.zeros_like(imgBlur)
    cv2.drawContours(Contorno_Fill, [imgEdge], 0, (255,255,255), cv2.FILLED)
    cv2.drawContours(Contorno, [imgEdge], 0, (255,255,255), 3)
    Contorno_Fill = cv2.bitwise_and(cv2.bitwise_not(Contorno),Contorno_Fill)

    #Segmentacion de la imagen
    return cv2.bitwise_and(imgBlur, Contorno_Fill)

def features21th(imgMasked, imgBlur, imgEdge, vector):# Obtencion 21 primeras caracteristicas
    y,x=imgBlur.shape[:2]
    #Filtro Canny
    Edge = feature.canny(imgMasked,sigma = 1).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    Edge = cv2.dilate(Edge, kernel, iterations = 2)

    #Threshold THblack
    _,shadows = cv2.threshold(imgMasked,115,255,cv2.THRESH_BINARY)

    #AND - THblack, Canny
    AND_operation = cv2.bitwise_and(shadows,Edge)*255

    #Obtener el contorno y su version rellena sin borde
    Contorno_Fill   = np.zeros_like(imgBlur)
    Contorno        = np.zeros_like(imgBlur)
    cv2.drawContours(Contorno_Fill, [imgEdge], 0, (255,255,255), cv2.FILLED)
    cv2.drawContours(Contorno, [imgEdge], 0, (255,255,255), 3)
    Contorno_Fill = cv2.bitwise_and(cv2.bitwise_not(Contorno),Contorno_Fill)

    #OR - AND, Contour
    Contorno_Detail = cv2.bitwise_or(Contorno, AND_operation)

    #Mascara sin los detalles del contorno
    Contorno_Fill_Edge = cv2.bitwise_and(cv2.bitwise_not(Contorno_Detail),Contorno_Fill)
    Contorno_Fill_Edge = cv2.erode(Contorno_Fill_Edge, kernel, iterations = 2)

    #Calculo de la transformada de distancia, radio y coordenadas para generar el circulo de la palma
    D_Transform = cv2.distanceTransform(Contorno_Fill, distanceType = cv2.DIST_L1, maskSize = 3)
    R_DT_1 = np.max(D_Transform)
    Coor_DT_1 = (np.argmax(D_Transform)%x,np.argmax(D_Transform)//x)

    #Circulos de la palma
    Circle = cv2.circle(np.zeros_like(imgBlur), Coor_DT_1, int(R_DT_1), 255, -1) 
    Circle_max = cv2.circle(np.zeros_like(imgBlur), Coor_DT_1, int(R_DT_1*1.5), 255, -1) 

    #Crear un muestreo de n puntos con el circulo mascara respecto la imagen original, lo cual dara indicios de en que lugar se encuentra la palma y los dedos
    Angle = np.linspace(0,360,1000)
    Radial_Sample = np.zeros_like(Angle)
    Index_y = np.clip((int(R_DT_1*1.5)*np.cos(Angle*np.pi/180)+Coor_DT_1[1]).astype(int),0,y-1)
    Index_x = np.clip((int(R_DT_1*1.5)*np.sin(Angle*np.pi/180)+Coor_DT_1[0]).astype(int),0,x-1)
    for i in range(0,len(Radial_Sample)):
            Radial_Sample[i] = Contorno_Fill[Index_y[i]][Index_x[i]]

    #Ajuste para comenzar el arreglo en el primer punto minimo
    Shift = np.argmin(Radial_Sample)
    Radial_Sample = np.roll(Radial_Sample,-Shift)
    Radial_Sample[-1] = 0

    #Realizar la transformada de distancia para identificar el punto central de los dedos y palma, ademas de su anchura
    D_Transform_Samples = cv2.distanceTransform(Radial_Sample.astype(np.uint8), distanceType = cv2.DIST_L1, maskSize = 3)
    D_T_Samples_R = np.max(D_Transform_Samples).astype(int)
    D_T_Samples_C = np.argmax(D_Transform_Samples).astype(int)

    #Ajustar los arreglos para que siempre empiecen con el punto maximo, correspondiente a la palma
    Radial_Sample = np.roll(Radial_Sample,D_T_Samples_R+len(Angle)-D_T_Samples_C)
    Index_y = np.roll(Index_y,D_T_Samples_R+len(Angle)-D_T_Samples_C-Shift)
    Index_x = np.roll(Index_x,D_T_Samples_R+len(Angle)-D_T_Samples_C-Shift)
    D_Transform_Samples = np.roll(D_Transform_Samples,D_T_Samples_R+len(Angle)-D_T_Samples_C)
    D_Transform_Samples = D_Transform_Samples.flatten()
    Angle = np.roll(Angle,D_T_Samples_R+len(Angle)-D_T_Samples_C-Shift)

    #Encontrar el indice del circulo en el que se encuentran los dedos y la palma, también su radio
    Peak_Center = signal.find_peaks(D_Transform_Samples, height = 10, width = 5)
    Peak_Center = Peak_Center[0].astype(int)
    if(len(Peak_Center)>6):
            Peak_Center = Peak_Center[:6]
    Peak_Radius = D_Transform_Samples[Peak_Center].astype(int)

    #Puntos centra, y extremos de los dedos
    Point_1 = np.vstack((Index_x[Peak_Center-Peak_Radius], Index_y[Peak_Center-Peak_Radius])).T
    Point_2 = np.vstack((Index_x[Peak_Center+Peak_Radius], Index_y[Peak_Center+Peak_Radius])).T
    Point_Center = np.vstack((Index_x[Peak_Center], Index_y[Peak_Center])).T
    Angle_Points = Angle[Peak_Center]

    FingerSize_1 = np.zeros_like(Peak_Center).astype(float)
    FingerSize_Inner_1 = np.zeros_like(Peak_Center).astype(float)
    #Lineas de los dedos y palma
    for i in range(0,len(Peak_Center)):
            FingerSize_1[i] = np.sqrt((Point_1[i][0]-Point_2[i][0])**2+(Point_1[i][1]-Point_2[i][1])**2)/R_DT_1
            FingerSize_Inner_1[i] = 1.4


    Point_Finger_Ext = np.zeros_like(Point_1)
    #Lineas de los dedos y palma
    m_img = y/x

    for i in range(1,len(Peak_Center)):
                    
            #Hallar pendiente ortogonal a la recta de los dedos (Para tener X = m*(y-y1)+x1)
            m = -(Point_1[i][1]-Point_2[i][1])/(Point_1[i][0]-Point_2[i][0])

            if(m>m_img):
                    y_finger = np.arange(Point_Center[i][1],y)
                    x_finger = m*(y_finger-Point_Center[i][1])+Point_Center[i][0]
                    x_overload_upper = np.where(x_finger >=  x)[0]
                    x_overload_lower = np.where(x_finger <=  0)[0]
                    
                    if(len(x_overload_upper)>0):
                            x_finger = x_finger[:x_overload_upper[0]-1]
                            y_finger = y_finger[:x_overload_upper[0]-1]
                    if(len(x_overload_lower)>0):
                            x_finger = x_finger[:x_overload_upper[0]-1]
                            y_finger = y_finger[:x_overload_upper[0]-1]

            else:
                    y_finger = np.arange(0,Point_Center[i][1])
                    x_finger = m*(y_finger-Point_Center[i][1])+Point_Center[i][0]
                    x_overload_upper = np.where(x_finger >=  x)[0]
                    x_overload_lower = np.where(x_finger <=  0)[0]

                    if(len(x_overload_upper)>0):
                            x_finger = x_finger[x_overload_upper[-1]+1:]
                            y_finger = y_finger[x_overload_upper[-1]+1:]
                    if(len(x_overload_lower)>0):
                            x_finger = x_finger[x_overload_lower[-1]+1:]
                            y_finger = y_finger[x_overload_lower[-1]+1:]

            x_samples = np.zeros_like(x_finger)

            #Hallar el punto externo del dedo que coincide con la recta proyectada ortogonal a la obtenida de los dedos
            for j in range(0,len(x_finger)):
                    x_samples[j] = Contorno_Fill[y_finger[j].astype(int)][x_finger[j].astype(int)]

                    if(m>m_img):
                            if(x_samples[j] == 0):
                                    break
                    else:   
                            if(x_samples[j] == 255):
                                    break

            Point_Finger_Ext[i] = [x_finger[j].astype(int),y_finger[j].astype(int)]

    FingerSize_Ext_1 = np.zeros_like(Peak_Center).astype(float)
    for i in range(1,len(Peak_Center)):
            FingerSize_Ext_1[i] = np.sqrt((Point_Center[i][0]-Point_Finger_Ext[i][0])**2+(Point_Center[i][1]-Point_Finger_Ext[i][1])**2)/R_DT_1


    #Transformada de distancia para verificar si existen dedos encogidos
    D_Transform_Aux = cv2.distanceTransform(Contorno_Fill_Edge, distanceType = cv2.DIST_L1, maskSize = 3)
    R_DT_Aux = np.max(D_Transform_Aux)
    Coor_DT_Aux = (np.argmax(D_Transform_Aux)%x,np.argmax(D_Transform_Aux)//x)

    #Segmentacion de los dedos compleja
    tol = 0.2
    if(not(np.abs(Coor_DT_1[0]-Coor_DT_Aux[0])/Coor_DT_1[0]<tol and np.abs(Coor_DT_1[1]-Coor_DT_Aux[1])/Coor_DT_1[1]<tol) and len(Peak_Center)<6):
            #Umbralizar para segmentar palma y dedos
            _,shadows_2 = cv2.threshold((D_Transform_Aux*255/np.max(R_DT_Aux)).astype(np.uint8),50,255,cv2.THRESH_BINARY)

            #Encontrar el contorno mas grande (Palma)
            c_Var = cv2.findContours(shadows_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c_Var = c_Var[0] if len(c_Var)  ==  2 else c_Var[1]
            C_var = max(c_Var, key = cv2.contourArea)
            
            Contorno_Palma = np.zeros_like(imgBlur)
            cv2.drawContours(Contorno_Palma, [C_var], 0, (255,255,255), cv2.FILLED)
            
            #Mascara para obtener los dedos
            Mask_Dedos_1 = cv2.bitwise_and(Circle, cv2.bitwise_not(Contorno_Palma))
            Dedos_1 = cv2.bitwise_and(Contorno_Fill_Edge, Mask_Dedos_1)

            #Transformada de distancia de los dedos
            Dedos_1 = cv2.distanceTransform(Dedos_1, distanceType = cv2.DIST_L1, maskSize = 3)

            # Transformada de distancia - Erosionada
            Dedos_1 = cv2.erode(Dedos_1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)), iterations = 10)
            Dedos_1 = cv2.dilate(Dedos_1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)), iterations = 10)
            _,Dedos_1 = cv2.threshold((Dedos_1*255/np.max(Dedos_1)).astype(np.uint8),80,255,cv2.THRESH_BINARY)

            #Crear un muestreo de n puntos con el circulo mascara respecto la imagen original, lo cual darA indicios de en que lugar se encuentran los dedos
            Radial_Sample_2 = np.zeros_like(Angle)
            Index_y_2 = (int(R_DT_1*0.9)*np.cos(Angle*np.pi/180)+Coor_DT_1[1]).astype(int)
            Index_x_2 = (int(R_DT_1*0.9)*np.sin(Angle*np.pi/180)+Coor_DT_1[0]).astype(int)

            for i in range(0,len(Radial_Sample_2)):
                    Radial_Sample_2[i] = Dedos_1[Index_y_2[i]][Index_x_2[i]]

            #Ajuste para comenzar el arreglo en el primer punto minimo
            Radial_Sample_2[-1] = 0

            #Realizar la transformada de distancia para identificar el punto central de los dedos y palma, ademas de su anchura
            D_Transform_Samples_2 = cv2.distanceTransform(Radial_Sample_2.astype(np.uint8), distanceType = cv2.DIST_L1, maskSize = 3)
            D_T_Samples_2_R = np.max(D_Transform_Samples_2).astype(int)
            D_T_Samples_2_C = np.argmax(D_Transform_Samples_2).astype(int)


            #Ajustar los arreglos para que siempre empiecen con el punto mAximo, correspondiente a la palma
            D_Transform_Samples_2 = D_Transform_Samples_2.flatten()

            #Encontrar el indice del circulo en el que se encuentran los dedos y la palma, también su radio
            Peak_Center_2 = signal.find_peaks(D_Transform_Samples_2, height = 10, width = 5)
            Peak_Center_2 = Peak_Center_2[0].astype(int)
            if(len(Peak_Center_2)>6-len(Peak_Center)):
                    Peak_Center_2 = Peak_Center_2[:6-len(Peak_Center)]

            Peak_Radius_2 = D_Transform_Samples_2[Peak_Center_2].astype(int)

            Point_1_Int = np.vstack((Index_x_2[Peak_Center_2-Peak_Radius_2], Index_y_2[Peak_Center_2-Peak_Radius_2])).T
            Point_2_Int = np.vstack((Index_x_2[Peak_Center_2+Peak_Radius_2], Index_y_2[Peak_Center_2+Peak_Radius_2])).T
            Point_Center_Int = np.vstack((Index_x_2[Peak_Center_2], Index_y_2[Peak_Center_2])).T
            Angle_Points_Int = Angle[Peak_Center_2]
            FingerSize_2 = np.zeros_like(Peak_Center_2).astype(float)
            FingerSize_Ext_2 = np.zeros_like(Peak_Center_2).astype(float)
            FingerSize_Inner_2 = np.zeros_like(Peak_Center_2).astype(float)

            for i in range(0,len(Peak_Center_2)):
                    FingerSize_2[i] = np.sqrt((Point_1_Int[i][0]-Point_2_Int[i][0])**2+(Point_1_Int[i][1]-Point_2_Int[i][1])**2)/R_DT_1
                    FingerSize_Ext_2[i] = np.sqrt((Point_Center_Int[i][0]-Coor_DT_1[0])**2+(Point_Center_Int[i][1]-Coor_DT_1[1])**2)/R_DT_1
                    FingerSize_Inner_2[i] = 0.9
    

    #Obtener el angulo tomando como referencia el quinto dedo
    if(not(np.abs(Coor_DT_1[0]-Coor_DT_Aux[0])/Coor_DT_1[0]<tol and np.abs(Coor_DT_1[1]-Coor_DT_Aux[1])/Coor_DT_1[1]<tol) and len(Peak_Center)<6):
            Angle_All = np.hstack((Angle_Points,Angle_Points_Int))
            FingerSize_All = np.hstack((FingerSize_1[1:],FingerSize_2))
            FingerSize_Ext_All = np.hstack((FingerSize_Ext_1[1:],FingerSize_Ext_2))
            FingerSize_Inner_All = np.hstack((FingerSize_Inner_1[1:],FingerSize_Inner_2))
    else:
            Angle_All = Angle_Points
            FingerSize_All = FingerSize_1[1:]
            FingerSize_Ext_All = FingerSize_Ext_1[1:]
            FingerSize_Inner_All = FingerSize_Inner_1[1:]

    #Ajuste del orden de los dedos y angulo de referencia
    print(len(Angle_All))
    Angle_All = Angle_All[1:]-Angle_All[0]
    Min_angle_ref = np.argmin(Angle_All)
    Angle_All = Angle_All-Angle_All[Min_angle_ref]

    #Indices para ordenar los dedos 
    Sort_Index = np.argsort(Angle_All)

    #Ordenar los arreglos de informacion en base al orden de los dedos
    Angle_All = np.sort(Angle_All)

    FingerSize_All = FingerSize_All[Sort_Index]
    FingerSize_Ext_All = FingerSize_Ext_All[Sort_Index]
    FingerSize_Inner_All = FingerSize_Inner_All[Sort_Index]

    #Numero de dedos
    vector[0] = len(Angle_All)

    #Angulo relativo de los dedos (Grados °)
    vector[1:len(Angle_All)] = Angle_All[1:]

    #Ancho de los dedos normalizado
    vector[5:5+len(FingerSize_All)] = FingerSize_All

    #Tamano de los dedos interno normalizado
    vector[10:10+len(FingerSize_Inner_All)] = FingerSize_Inner_All

    #Tamano de los dedos externo normalizado
    vector[15:15+len(FingerSize_Ext_All)] = FingerSize_Ext_All

def HUMoments(imgEdge, vector): # Momentos de Hu
    HuMoments =cv2.HuMoments(cv2.moments(imgEdge))
    idx = 0
    for moment in HuMoments:
        vector[20 + idx] = moment[0]
        idx += 1

def getFeatures(img): # Vector de caracteristicas
    imgBlur = blurImg(img) # Suavizado del histograma
    imgOtsu = otsu(imgBlur) # Obtencion del umbral por otsu
    imgEdge = edge(imgOtsu) # Obtener borde
    perimeter = cv2.arcLength(imgEdge, 1) # Perimetro
    area = cv2.contourArea(imgEdge) # Area
    imgMasked = segmentation(imgBlur, imgEdge) # Segmentacion de la imagen
    features = np.zeros(29).astype(float) # Vector 1x29
    features21th(imgMasked, imgBlur, imgEdge, features) # Extraccion de las 21 primeras caracteristicas
    HUMoments(imgEdge, features) # Momentos de Hu
    features[27] = np.pi * area/(perimeter*perimeter) # Redondez
    features[28] = perimeter*perimeter/area # Compacidad
    return features

path=os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(path, 'Project_Data_Fingers.csv')


with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(featuresVector)



    Clases_Length=len(os.listdir(os.path.join(path, 'Images_Fingers')))

    for i in range(1,Clases_Length+1):
            Data_length=len(os.listdir(os.path.join(os.path.join(path, 'Images_Fingers'),str(i))))
            
            for j in range(1,Data_length+1):
                
                Image_path=os.path.join(os.path.join(os.path.join(path, 'Images_Fingers'),str(i)),str(j)+'.jpg')
                print(Image_path)
                
                Features_Vec = np.zeros((30)).astype(float)
                img = cv2.imread(Image_path,1)
                Features_Vec[:29]=getFeatures(img)
                Features_Vec[29]=i
                writer.writerow(Features_Vec)










