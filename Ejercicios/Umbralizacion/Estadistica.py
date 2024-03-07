import cv2
import os

from IPython.display import Image
import numpy as np
from matplotlib import pyplot as plt

Test  =   cv2.imread(os.path.dirname(__file__)+'/Images/Img_Outdoor_1.png',0)
Hist     =   cv2.calcHist([Test],[0],None,[2**8],[0,2**8])
H,W=Test.shape[:2]


k1=1
k2=k1
    
plt.stem(cv2.calcHist([Test],[0],None,[2**8],[0,2**8]))
plt.show()

a=0
b=255

for i in range(1,5):

    mean=np.mean(Test[np.logical_and(Test<b,Test>a)])
    std=np.std(Test[np.logical_and(Test<b,Test>a)])

    mean_val1=np.mean(Test[np.logical_and(Test<(mean-k1*std),Test>a)])
    mean_val2=np.mean(Test[np.logical_and(Test>(mean+k2*std),Test<b)])

    Test[np.logical_and(Test<(mean-k1*std),Test>a)]=mean_val1
    Test[np.logical_and(Test>(mean+k2*std),Test<b)]=mean_val2

    a=mean-k1*std+1
    b=mean+k2*std-1



    # Test[np.logical_and(Test<mean,Test>a)]=np.mean(Test[np.logical_and(Test<mean,Test>a)])
    # Test[np.logical_and(Test>(mean+1),Test<b)]=np.mean(Test[np.logical_and(Test>(mean+1),Test<b)])

    plt.stem(cv2.calcHist([Test],[0],None,[2**8],[0,2**8]))
    plt.show()

Test[np.logical_and(Test<mean,Test>a)]=mean_val1
Test[np.logical_and(Test>(mean+1),Test<b)]=mean_val2

plt.stem(cv2.calcHist([Test],[0],None,[2**8],[0,2**8]))
plt.show()

cv2.imshow('Test',Test)

cv2.waitKey(0)
cv2.destroyAllWindows()