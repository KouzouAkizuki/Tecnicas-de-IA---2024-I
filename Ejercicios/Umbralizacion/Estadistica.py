import cv2
import os

from IPython.display import Image
import numpy as np
from matplotlib import pyplot as plt

Test  =   cv2.imread(os.path.dirname(__file__)+'/Images/Img_Outdoor_1.png',0)
Hist     =   cv2.calcHist([Test],[0],None,[2**8],[0,2**8])
H,W=Test.shape[:2]
