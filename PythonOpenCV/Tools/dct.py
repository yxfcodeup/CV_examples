import os
import sys
import math
from functools import reduce
#External Libs
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from basics import cv2PIL

PI = math.pi


def func_c(u , n) :
    if 0 == u :
        return (1/n) ** 0.5
    else :
        return (2/n) ** 0.5
    return False

def dct_2dim(array_2d) :
    u = array_2d.shape[0]
    v = array_2d.shape[1]
    res = np.zeros((u , v))
    for i in range(u) :
        for j in range(v) :
            cu = func_c(i , u)
            cv = func_c(j , v)
            res[i , j] = cu * cv * array_2d[i , j] \
                    * math.cos((i+0.5) * PI * i / u) \
                    * math.cos((i+0.5) * PI * j / v)
    return res

def dct(gray_img) :
    N = gray_img.shape[0]
    A = np.zeros((N , N))   #变换矩阵
    for i in range(N) :
        for j in range(N) :
            coeff = math.sqrt(1/N) if 0==i else math.sqrt(2/N)  #变换系数
            A[i , j] = coeff * math.cos((j+0.5) * PI * i / N)
    dct_y = np.dot(np.dot(A , gray_img) , A.T)  #A*f*A.T
    return dct_y



if "__main__" == __name__ :
    img1 = cv2.imread("./t1.jpg")
    img1 = cv2.resize(img1 , (128 , 128))
    gray1 = np.float32(cv2.cvtColor(img1 , cv2.COLOR_BGR2GRAY))
    dct1 = cv2.dct(gray1)
    dct2 = dct_2dim(gray1)
    dct3 = dct(gray1)
    print("\ndct1:\n" , dct1[:5 , :5])
    print("\ndct2:\n" , dct2[:5 , :5])
    print("\ndct3:\n" , dct3[:5 , :5])
