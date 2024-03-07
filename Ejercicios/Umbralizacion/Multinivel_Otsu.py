import cv2
import os

from IPython.display import Image
import numpy as np
from matplotlib import pyplot as plt

Test  =   cv2.imread(os.path.dirname(__file__)+'/Images/Img_Outdoor_1.png',0)
Hist     =   cv2.calcHist([Test],[0],None,[2**8],[0,2**8])
H,W=Test.shape[:2]


Hist=Hist/(H*W) #Normaliza el Histograma (Funcion de probabilidad)
mean=sum(x*i for i,x in enumerate(Hist)) # Media de la funcion de probabilidad

for Umbral in range(1,255):

    P_r1=sum(Hist[0:Umbral]) # Probabilidad acumulativa de la region 1
    P_r2=sum(Hist[Umbral:])  # Probabilidad acumulativa de la region 2


    # print('Region #1: '+str(P_r1))
    # print('Region #2: '+str(P_r2))

    # print('Total: '+str(P_r1+P_r2))

    mean_r1=sum(x*i for i,x in enumerate(Hist[0:Umbral]))/(P_r1)            # Media de la region 1
    mean_r2=sum(x*(i+Umbral) for i,x in enumerate(Hist[Umbral:]))/(P_r2)    # Media de la region 2

    # print('Media #1: '+str(mean_r1))
    # print('Media #2: '+str(mean_r2))
    # print('Media Total: '+str(mean))

    # print(mean_r2*P_r2+mean_r1*P_r1) #Propiedad para confirmar las medias

    Variance_Global=sum(x*(i-mean)**2 for i,x in enumerate (Hist))          # Varianza Global
    Variance_BW_Class=P_r1*(mean_r1-mean)**2+P_r2*(mean_r2-mean)**2         # Varianza entre Regiones
    Coef=Variance_BW_Class/Variance_Global                                  # Coeficiente
    
    if(np.isnan(Coef)[0]):
        Coef=[0]

    if Umbral==1: 
        Coeff_list=Coef[0]
    else:
         Coeff_list=np.append(Coeff_list,Coef)


    







plt.plot(Coeff_list)
plt.show()