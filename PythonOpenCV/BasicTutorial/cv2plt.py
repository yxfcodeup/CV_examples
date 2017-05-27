import os
import sys
#External Libs
import cv2
import numpy as np
from PIL import Image
from pylab import *
from matplotlib import pyplot as plt

img1 = cv2.imread("./ss.jpg")
img2 = Image.open("./ss.jpg").convert("RGB")
img3 = np.array(img2)
img3 = img3[: , : , ::-1].copy()
img4 = np.zeros(img1.shape , img1.dtype)
img4[: , : , 0] = img1[: , : , 2]
img4[: , : , 1] = img1[: , : , 1]
img4[: , : , 2] = img1[: , : , 0]
#gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
ims = plt.imshow(img4)
plt.show()
