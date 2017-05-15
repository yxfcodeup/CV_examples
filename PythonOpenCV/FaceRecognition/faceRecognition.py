import os
import sys
#External Libs
import cv2
import numpy as np
from PIL import Image
from pylab import *
from matplotlib import pyplot as plt

"""
haarcascade_frontalface_default.xml是opencv针对人脸已经训练好的模型数据
"""

img1 = cv2.imread("./ScarlettJohansson.jpg")
gray_img = cv2.cvtColor(img1 , cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(
        gray_img , 
        scaleFactor = 1.15 ,
        minNeighbors = 5 , 
        minSize = (5 , 5)
        )

for (x , y , w , h) in faces :
    cv2.rectangle(gray_img , (x , y) , (x+w , y+h) , (0 , 255 , 0) , 2)

cv2.imwrite("out.jpg" , gray_img)
"""
img4 = np.zeros(gray_img.shape , gray_img.dtype)
img4[: , : , 0] = img1[: , : , 2]
img4[: , : , 1] = img1[: , : , 1]
img4[: , : , 2] = img1[: , : , 0]
ims = plt.imshow(img4)
plt.show()
"""
