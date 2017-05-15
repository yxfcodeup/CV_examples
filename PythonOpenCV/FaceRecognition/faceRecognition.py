import os
import sys
#External Libs
import cv2

img = cv2.imread("./ScarlettJohansson.jpg")
gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
cv2.imshow("ScarlettJohansson" , img)
