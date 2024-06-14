from skimage import feature, filters
from scipy import signal
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

featuresVector = ['Numero de dedos', 'Angulo indice', 'Angulo corazon', 'Angulo anular', 'Angulo menique',
                  'Ancho pulgar', 'Ancho indice', 'Ancho corazon', 'Ancho anular', 'Ancho menique',
                  'Pulgar extendido', 'Indice extendido', 'Corazon extendido', 'Anular extendido', 'Menique extendido',
                  'Pulgar contraido', 'Indice contraido', 'Corazon contraido', 'Anular contraido', 'Menique contraido']

############################### Proceso de segmentacion de la imagen ###############################
#### Proceso de segmentacion de la imagen ####
Imagen = [Indoor_G, Pattern_G, Green_G]
# Imagen=[Indoor_P, Pattern_P, Green_P]

#Segmentacion de la imagen
Seg = [Segmentation(a) for a in Imagen]
Image_Masked =  [cv2.bitwise_and(Seg[a][2], Seg[a][2], mask=Seg[a][0]) for a in range(len(Imagen))]

#Contorno de los dedos
Finger_Contours = [Finger_Contour(Seg[a][0], Seg[a][2], Seg[a][1]) for a in range(len(Imagen))]

Characteristics=[]
#Referencia del tiempo de ejecucion
ref=time.time()

for i in range(3):

        ############################## Extraccion de caracteristicas ###############################

        #Calculo de la transformada de distancia, radio y coordenadas para generar el circulo de la palma
        D_Transform=cv2.distanceTransform(Seg[i][0], distanceType=cv2.DIST_L1, maskSize=3)
        R_DT_1=np.max(D_Transform)
        Coor_DT_1=(np.argmax(D_Transform)%Seg[i][3],np.argmax(D_Transform)//Seg[i][3])

        #Circulos de la palma
        Circle=cv2.circle(np.zeros_like(Seg[i][2]), Coor_DT_1, int(R_DT_1), 255, -1) 
        Circle_max=cv2.circle(np.zeros_like(Seg[i][2]), Coor_DT_1, int(R_DT_1*1.5), 255, -1) 

        #Crear un muestreo de n puntos con el circulo mascara respecto la imagen original, lo cual dara indicios de en que lugar se encuentra la palma y los dedos
        Angle=np.linspace(0,360,1000)
        Radial_Sample=np.zeros_like(Angle)
        Index_y=np.clip((int(R_DT_1*1.5)*np.cos(Angle*np.pi/180)+Coor_DT_1[1]).astype(int),0,Seg[i][4]-1)
        Index_x=np.clip((int(R_DT_1*1.5)*np.sin(Angle*np.pi/180)+Coor_DT_1[0]).astype(int),0,Seg[i][3]-1)
        
        # print(type(Seg[i][0][Index_y[0],Index_x[0]]))
        for k in range(0,len(Radial_Sample)):
                Radial_Sample[k]=Seg[i][0][Index_y[k]][Index_x[k]]

        #Ajuste para comenzar el arreglo en el primer punto minimo
        Shift=np.argmin(Radial_Sample)
        Radial_Sample=np.roll(Radial_Sample,-Shift)
        Radial_Sample[-1]=0

        # Realizar la transformada de distancia para identificar el punto central de los dedos y palma, ademas de su anchura
        D_Transform_Samples=cv2.distanceTransform(Radial_Sample.astype(np.uint8), distanceType=cv2.DIST_L1, maskSize=3)
        D_T_Samples_R=np.max(D_Transform_Samples).astype(int)
        D_T_Samples_C=np.argmax(D_Transform_Samples).astype(int)

        #Ajustar los arreglos para que siempre empiecen con el punto maximo, correspondiente a la palma
        Radial_Sample=np.roll(Radial_Sample,D_T_Samples_R+len(Angle)-D_T_Samples_C)
        Index_y=np.roll(Index_y,D_T_Samples_R+len(Angle)-D_T_Samples_C-Shift)
        Index_x=np.roll(Index_x,D_T_Samples_R+len(Angle)-D_T_Samples_C-Shift)
        D_Transform_Samples=np.roll(D_Transform_Samples,D_T_Samples_R+len(Angle)-D_T_Samples_C)
        D_Transform_Samples=D_Transform_Samples.flatten()
        Angle=np.roll(Angle,D_T_Samples_R+len(Angle)-D_T_Samples_C-Shift)

        #Encontrar el indice del circulo en el que se encuentran los dedos y la palma, tambien su radio
        Peak_Center=signal.find_peaks(D_Transform_Samples, height=10, width=5)
        Peak_Center=Peak_Center[0].astype(int)
        if(len(Peak_Center)>6):
                Peak_Center=Peak_Center[:6]
        Peak_Radius=D_Transform_Samples[Peak_Center].astype(int)

        #Puntos centra, y extremos de los dedos
        Point_1 = np.vstack((Index_x[Peak_Center-Peak_Radius], Index_y[Peak_Center-Peak_Radius])).T
        Point_2 = np.vstack((Index_x[Peak_Center+Peak_Radius], Index_y[Peak_Center+Peak_Radius])).T
        Point_Center = np.vstack((Index_x[Peak_Center], Index_y[Peak_Center])).T
        Angle_Points = Angle[Peak_Center]

        FingerSize_1=np.zeros_like(Peak_Center).astype(float)
        FingerSize_Inner_1=np.zeros_like(Peak_Center).astype(float)
        #Lineas de los dedos y palma
        xd = Imagen[i].astype(np.uint8)
        for k in range(0,len(Peak_Center)):
                FingerSize_1[k]=np.sqrt((Point_1[k][0]-Point_2[k][0])**2+(Point_1[k][1]-Point_2[k][1])**2)/R_DT_1
                FingerSize_Inner_1[k]=1.4
                xd=cv2.line(xd, pt1=Point_2[k], pt2=Point_1[k], color=(0, 255, 0), thickness=20)


        Point_Finger_Ext=np.zeros_like(Point_1)
        #Lineas de los dedos y palma
        m_img=Seg[i][4]/Seg[i][3]

        
        for k in range(1,len(Peak_Center)):
                        
                #Hallar pendiente ortogonal a la recta de los dedos (Para tener X=m*(y-y1)+x1)
                m = -(Point_1[k][1]-Point_2[k][1])/(Point_1[k][0]-Point_2[k][0])


                if(m>m_img):
                        y_finger=np.arange(Point_Center[k][1],Seg[i][4])
                        x_finger = m*(y_finger-Point_Center[k][1])+Point_Center[k][0]
                        x_overload_upper = np.where(x_finger >= Seg[k][3])[0]
                        x_overload_lower = np.where(x_finger <= 0)[0]
                        
                        if(len(x_overload_upper)>0):
                                x_finger = x_finger[:x_overload_upper[0]-1]
                                y_finger = y_finger[:x_overload_upper[0]-1]
      
                        if(len(x_overload_lower)>0):
                                x_finger = x_finger[:x_overload_upper[0]-1]
                                y_finger = y_finger[:x_overload_upper[0]-1]


                else:
                        y_finger=np.arange(0,Point_Center[k][1])
                        x_finger = m*(y_finger-Point_Center[k][1])+Point_Center[k][0]
                        x_overload_upper = np.where(x_finger >= Seg[i][3])[0]
                        x_overload_lower = np.where(x_finger <= 0)[0]

                        if(len(x_overload_upper)>0):
                                x_finger = x_finger[x_overload_upper[-1]+1:]
                                y_finger = y_finger[x_overload_upper[-1]+1:]
                        if(len(x_overload_lower)>0):
                                x_finger = x_finger[x_overload_lower[-1]+1:]
                                y_finger = y_finger[x_overload_lower[-1]+1:]

                x_samples=np.zeros_like(x_finger)

                #Hallar el punto externo del dedo que coincide con la recta proyectada ortogonal a la obtenida de los dedos
                for j in range(0,len(x_finger)):
                        x_samples[j]=Seg[i][0][y_finger[j].astype(int)][x_finger[j].astype(int)]

                        if(m>m_img):
                                if(x_samples[j]==0):
                                        break
                        else:   
                                if(x_samples[j]==255):
                                        break

                Point_Finger_Ext[k]=[x_finger[j].astype(int),y_finger[j].astype(int)]

        FingerSize_Ext_1=np.zeros_like(Peak_Center).astype(float)
        for k in range(1,len(Peak_Center)):
                xd=cv2.line(xd, Point_Center[k], Point_Finger_Ext[k], (0, 255, 0), thickness=20)
                FingerSize_Ext_1[k]=np.sqrt((Point_Center[k][0]-Point_Finger_Ext[k][0])**2+(Point_Center[k][1]-Point_Finger_Ext[k][1])**2)/R_DT_1


        plt.figure(1)
        plt.subplot(15,3,i+1), plt.imshow(Imagen[i]), plt.title('Original '+title2[i])
        plt.axis('off')
        plt.subplot(15,3,i+4), plt.imshow(Image_Masked[i] ,cmap=plt.cm.gray), plt.title('Segmentation '+title2[i], fontsize = 8)
        plt.axis('off')
        plt.subplot(15,3,i+7), plt.imshow(D_Transform ,cmap=plt.cm.gray), plt.title('Transformada de distancia '+title2[i], fontsize = 5)
        plt.axis('off')
        plt.subplot(15,3,i+10), plt.plot(Radial_Sample), plt.title('Muestreo del circulo '+title2[i], fontsize = 5)
        plt.subplot(15,3,i+13), plt.plot(D_Transform_Samples), plt.title('Transformada de distancia - 1D '+title2[i], fontsize = 5)
        plt.subplot(15,3,i+16), plt.imshow(xd ,cmap=plt.cm.gray), plt.title('Dedos identificados #1 ' +title2[i], fontsize = 4)
        plt.axis('off')


        #Transformada de distancia para verificar si existen dedos encogidos
        D_Transform_Aux=cv2.distanceTransform(Finger_Contours[i][1], distanceType=cv2.DIST_L1, maskSize=3)
        R_DT_Aux=np.max(D_Transform_Aux)
        Coor_DT_Aux=(np.argmax(D_Transform_Aux)%Seg[i][3],np.argmax(D_Transform_Aux)//Seg[i][3])

        #Segmentacion de los dedos compleja
        tol=0.2
        if(not(np.abs(Coor_DT_1[0]-Coor_DT_Aux[0])/Coor_DT_1[0]<tol and np.abs(Coor_DT_1[1]-Coor_DT_Aux[1])/Coor_DT_1[1]<tol) and len(Peak_Center)<6):
                #Umbralizar para segmentar palma y dedos
                _,shadows_2 = cv2.threshold((D_Transform_Aux*255/np.max(R_DT_Aux)).astype(np.uint8),50,255,cv2.THRESH_BINARY)

                #Encontrar el contorno mas grande (Palma)
                c_Var = cv2.findContours(shadows_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c_Var = c_Var[0] if len(c_Var) == 2 else c_Var[1]
                C_var = max(c_Var, key=cv2.contourArea)
                
                Contorno_Palma = np.zeros_like(Seg[i][2])
                cv2.drawContours(Contorno_Palma, [C_var], 0, (255,255,255), cv2.FILLED)

                #Mascara para obtener los dedos
                Mask_Dedos_1 = cv2.bitwise_and(Circle, cv2.bitwise_not(Contorno_Palma))
                Dedos_1 = cv2.bitwise_and(Finger_Contours[i][1], Mask_Dedos_1)

                #Transformada de distancia de los dedos
                Dedos_1=cv2.distanceTransform(Dedos_1, distanceType=cv2.DIST_L1, maskSize=3)

                # Transformada de distancia - Erosinada
                Dedos_1=cv2.erode(Dedos_1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)), iterations = 10)
                Dedos_1=cv2.dilate(Dedos_1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)), iterations = 10)
                _,Dedos_1 = cv2.threshold((Dedos_1*255/np.max(Dedos_1)).astype(np.uint8),80,255,cv2.THRESH_BINARY)

                #Crear un muestreo de n puntos con el circulo mascara respecto la imagen original, lo cual dara indicios de en que lugar se encuentran los dedos
                Radial_Sample_2=np.zeros_like(Angle)
                Index_y_2=(int(R_DT_1*0.9)*np.cos(Angle*np.pi/180)+Coor_DT_1[1]).astype(int)
                Index_x_2=(int(R_DT_1*0.9)*np.sin(Angle*np.pi/180)+Coor_DT_1[0]).astype(int)

                for k in range(0,len(Radial_Sample_2)):
                        Radial_Sample_2[k]=Dedos_1[Index_y_2[k]][Index_x_2[k]]

                #Ajuste para comenzar el arreglo en el primer punto minimo
                Radial_Sample_2[-1]=0

                #Realizar la transformada de distancia para identificar el punto central de los dedos y palma, ademas de su anchura
                D_Transform_Samples_2=cv2.distanceTransform(Radial_Sample_2.astype(np.uint8), distanceType=cv2.DIST_L1, maskSize=3)
                D_T_Samples_2_R=np.max(D_Transform_Samples_2).astype(int)
                D_T_Samples_2_C=np.argmax(D_Transform_Samples_2).astype(int)


                #Ajustar los arreglos para que siempre empiecen con el punto maximo, correspondiente a la palma
                D_Transform_Samples_2=D_Transform_Samples_2.flatten()

                #Encontrar el indice del circulo en el que se encuentran los dedos y la palma, tambien su radio
                Peak_Center_2=signal.find_peaks(D_Transform_Samples_2, height=10, width=5)
                Peak_Center_2=Peak_Center_2[0].astype(int)
                if(len(Peak_Center_2)>6-len(Peak_Center)):
                        Peak_Center_2=Peak_Center_2[:6-len(Peak_Center)]

                Peak_Radius_2=D_Transform_Samples_2[Peak_Center_2].astype(int)

                Point_1_Int=np.vstack((Index_x_2[Peak_Center_2-Peak_Radius_2], Index_y_2[Peak_Center_2-Peak_Radius_2])).T
                Point_2_Int=np.vstack((Index_x_2[Peak_Center_2+Peak_Radius_2], Index_y_2[Peak_Center_2+Peak_Radius_2])).T
                Point_Center_Int=np.vstack((Index_x_2[Peak_Center_2], Index_y_2[Peak_Center_2])).T
                Angle_Points_Int=Angle[Peak_Center_2]

                FingerSize_2=np.zeros_like(Peak_Center_2).astype(float)
                FingerSize_Ext_2=np.zeros_like(Peak_Center_2).astype(float)
                FingerSize_Inner_2=np.zeros_like(Peak_Center_2).astype(float)

                for k in range(0,len(Peak_Center_2)):
                        FingerSize_2[k]=np.sqrt((Point_1_Int[k][0]-Point_2_Int[k][0])**2+(Point_1_Int[k][1]-Point_2_Int[k][1])**2)/R_DT_1
                        FingerSize_Ext_2[k]=np.sqrt((Point_Center_Int[k][0]-Coor_DT_1[0])**2+(Point_Center_Int[k][1]-Coor_DT_1[1])**2)/R_DT_1
                        FingerSize_Inner_2[k]=0.9
                        xd=cv2.line(xd, Point_2_Int[k], Point_1_Int[k], (0, 255, 0), thickness=20)
                        xd=cv2.line(xd, Point_Center_Int[k], Coor_DT_1, (0, 255, 0), thickness=20)

        #Obtener el angulo tomando como referencia el quinto dedo
        if(len(Peak_Center)<6):
                Angle_All=np.hstack((Angle_Points,Angle_Points_Int))
                FingerSize_All=np.hstack((FingerSize_1[1:],FingerSize_2))
                FingerSize_Ext_All=np.hstack((FingerSize_Ext_1[1:],FingerSize_Ext_2))
                FingerSize_Inner_All=np.hstack((FingerSize_Inner_1[1:],FingerSize_Inner_2))
        else:
                Angle_All=Angle_Points
                FingerSize_All=FingerSize_1[1:]
                FingerSize_Ext_All=FingerSize_Ext_1[1:]
                FingerSize_Inner_All=FingerSize_Inner_1[1:]

        #Ajuste del orden de los dedos y angulo de referencia
        Angle_All=Angle_All[1:]-Angle_All[0]
        Min_angle_ref=np.argmin(Angle_All)
        Angle_All=Angle_All-Angle_All[Min_angle_ref]

        #Indices para ordenar los dedos 
        Sort_Index=np.argsort(Angle_All)

        #Ordenar los arreglos de informacion en base al orden de los dedos
        Angle_All=np.sort(Angle_All)
        FingerSize_All=FingerSize_All[Sort_Index]
        FingerSize_Ext_All=FingerSize_Ext_All[Sort_Index]
        FingerSize_Inner_All=FingerSize_Inner_All[Sort_Index]

        #Vector de caracteristicas
        Caracteristicas=np.zeros(20).astype(float)

        #Numero de dedos
        Caracteristicas[0]=len(Angle_All)

        #Angulo relativo de los dedos (Grados °)
        Caracteristicas[1:len(Angle_All)]=Angle_All[1:len(Angle_All)]

        #Ancho de los dedos normalizado
        Caracteristicas[5:5+len(FingerSize_All)]=FingerSize_All

        #Tamaño de los dedos interno normalizado
        Caracteristicas[10:10+len(FingerSize_Inner_All)]=FingerSize_Inner_All

        #Tamaño de los dedos externo normalizado
        Caracteristicas[15:15+len(FingerSize_Ext_All)]=FingerSize_Ext_All

        Characteristics.append(Caracteristicas)

for i in range(3):
        print(Characteristics[i])