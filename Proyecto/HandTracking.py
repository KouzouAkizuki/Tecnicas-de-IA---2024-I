import cv2
import time
import os
import numpy as np
import math
stabilize_highest_point = True
old_highest_point = (-1, -1)
full_frame = True
drawing_box = True

Mano        =   cv2.imread(os.path.dirname(__file__)+'/Images/Img_Indoor_1.png',1)

def detectHand(img, kernel_dim = (5,5)):
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(img, (3,3), 0)
    



    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        
    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, int(255 * 0.68), 255]))
    



    # Kernel for morphological transformation    
    kernel = np.ones(kernel_dim)
    
    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations = 2)
    erosion = cv2.erode(dilation, kernel, iterations = 2)    
       
    # Apply Gaussian Blur and Threshold (To clean)
    filtered = cv2.GaussianBlur(erosion, (3,3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)
    
    # cv2.imshow('filtered', filtered)
    # cv2.imshow('thresh - filtered', thresh - filtered)
    # cv2.imshow('thresh', thresh)
    
    try:   
        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
        
        # Find biggest contour
        contour = max(contours, key = lambda x: cv2.contourArea(x))
       
        return contour
    except:
        
        return np.zeros(0)

def findDefects(crop_image, contour):
    # Create bounding rectangle around the contour
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(crop_image, (x,y), (x+w,y+h), (255,0,255), 0)
    
    # Find convex hull
    hull = cv2.convexHull(contour)
    
    # Draw contour
    drawing = np.zeros(crop_image.shape, np.uint8)
    cv2.drawContours(drawing, [contour], -1, (0,0,255), 0)
    cv2.drawContours(drawing, [hull], -1, (0,255,0), 0)
    
    # Find convexity defects
    hull = cv2.convexHull(contour, returnPoints = False)
    defects = cv2.convexityDefects(contour, hull)
    
    return defects, drawing

def countDefects(defects, contour, crop_image):
    count_defects = 0
    for i in range(defects.shape[0]):
        # if(i == 0): print(defects[i,0])
        
        s, e, f, d = defects[i,0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
        
        # if angle < 90 draw a circle at the far point
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_image, start, 10, [255,0,0], -1)
            cv2.circle(crop_image, end, 10, [0,255,0], -1)
            cv2.circle(crop_image, far, 10, [0,0,255], -1)
            
        cv2.line(crop_image, start, end, [0,255,0], 2)
        
    return count_defects

def trackHighestPoint(defects, contour):
    # Tracking of the highest point detected
    highest_point = (1920, 1080)
    
    for i in range(defects.shape[0]):
        # if(i == 0): print(defects[i,0])
        
        s,e,f,d = defects[i,0]
        tmp_point = tuple(contour[s][0])
        
        if(tmp_point[1] < highest_point[1]): highest_point = tmp_point;
        
    return highest_point

def textDefects(frame, count_defects,  color = [255,0,255], debug_var = False):
    if(debug_var): print("Defects : ", count_defects)
   
    if count_defects == 0:
        cv2.putText(frame,"ZERO", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, color)
    elif count_defects == 1:
        cv2.putText(frame,"TWO", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, color)
    elif count_defects == 2:
        cv2.putText(frame, "THREE", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 2, color)
    elif count_defects == 3:
        cv2.putText(frame,"FOUR", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, color)
    elif count_defects == 4:
        cv2.putText(frame,"FIVE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, color)
    else:
        pass

contour_left = detectHand(Mano)
defects_left, drawing_left = findDefects(Mano, contour_left)
count_defects = countDefects(defects_left, contour_left, Mano)
highest_point = trackHighestPoint(defects_left, contour_left)

if(stabilize_highest_point):
    if( old_highest_point == (-1, -1)): old_highest_point = highest_point
    else:
        # Evaluate the magnitude of the difference
        diag_difference = np.linalg.norm(np.asarray(old_highest_point) - np.asarray(highest_point))
                
        # If the difference is bigger than a threshold then I actually moved my finger
        if(diag_difference >= 9.5): 
            # print("diag_difference = ", diag_difference)
            old_highest_point = highest_point
        else: highest_point = old_highest_point;
            
if(full_frame):
    highest_point = (highest_point[0], highest_point[1])
    cv2.circle(Mano, highest_point, 10, [255,0,255], -1)

    
# Print number of fingers
textDefects(Mano, count_defects,debug_var = False)
all_image_left = np.hstack((drawing_left, Mano))
   

cv2.imshow("Full Frame", Mano)


cv2.waitKey(0)
cv2.destroyAllWindows()