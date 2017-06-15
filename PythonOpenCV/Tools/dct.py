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

def dct2(gray_img) :
    u = gray_img.shape[0]
    v = gray_img.shape[1]
    res = np.zeros((u , v))
    coeff = np.zeros((u , v))   #变换系数
    for i in range(u) :
        for j in range(v) :
            coeff[i , j] = math.sqrt(1/n) if 0==j else math.sqrt(2/n)


def dct_1d(array_1d) :
    N = array_1d.shape[0]
    res = np.zeros((N , 1))
    coeff = np.zeros(N)
    for u in range(N) :
        for i in range(N) :
            res[i , 0] += array_1d[i] * math.cos((i+0.5) * PI * u / N)
        cfn = math.sqrt(1/N) if 0==u else math.sqrt(2/N)
        res[u , 0] *= cfn
    return res

def dct_2d(array_2d) :
    N1 = array_2d.shape[0]
    N2 = array_2d.shape[1]
    cfn1 = np.zeros(N1)
    cfn2 = np.zeros(N2)
    tmp = np.zeros((N1 , N2))
    res = np.zeros((N1 , N2))
    for u in range(N1) :
        cfn1[u] = math.sqrt(1/N1) if 0==u else math.sqrt(2/N1)
    for v in range(N2) :
        cfn1[v] = math.sqrt(1/N2) if 0==u else math.sqrt(2/N2)


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
    a1 = np.array([1,2,3,4] , np.float32)
    r1 = cv2.dct(a1)
    r2 = dct_1d(a1)
    print(r1)
    print(r2)
