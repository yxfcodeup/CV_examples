import os
import sys
#External Libs
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

#BGR <--> RGB
def cv2PIL(image , flag="BGR2RGB") :
    if "BGR2RGB" == flag :
        img = np.zeros(image.shape , image.dtype)
        img[: , : , 0] = image[: , : , 2]
        img[: , : , 1] = image[: , : , 1]
        img[: , : , 2] = image[: , : , 0]
        return img
    elif "RGB2BGR" == flag :
        img = np.array(image)
        img = img[: , : , ::-1].copy()
        return img
    else :
        print("flag must be 'BGR2RGB' or 'RGB2BGR'")
        return False

def histogram(cv_img , flag="GRAY") :
    if "GRAY" == flag :
        hist = cv2.calcHist([cv_img] ,  
                [0] ,           #使用的通道  
                None ,          #没有使用mask  
                [256] ,         #HistSize  
                [0.0,255.0])    #直方图柱的范围
        return hist
    elif "BGR" == flag :
        b , g , r = cv2.split(cv_img)
        def calcAndDrawHist(img , color) :
            hist = cv2.calcHist([img] , [0] , None , [256] , [0.0,255.0])
            min_val , max_val , min_loc, max_loc = cv2.minMaxLoc(hist)
            hist_img = np.zeros([256,256,3] , np.uint8)
            hpt = int(0.9 * 256) 
            for h in range(256) :
                intensity = int(hist[h] * hpt / max_val)
                cv2.line(hist_img , (h,256) , (h,256-intensity) , color)
            return hist_img
        hist_b = calcAndDrawHist(b , [255,0,0])
        hist_g = calcAndDrawHist(g , [0,255,0])
        hist_r = calcAndDrawHist(r , [0,0,255])
        return (hist_b , hist_g , hist_r)
    elif "MULTICOLOR" == flag :
        h = np.zeros((256 , 256 , 3))   #创建用于绘制直方图的全0图像
        bins = np.arange(256).reshape(256,1)    #直方图中各bin的顶点位置
        colors = [(255,0,0) , (0,255,0) , (0,0,255) ]   #BGR三种颜色
        for channel , color in enumerate(colors) :
            hist_item = cv2.calcHist([cv_img] , [channel] , None , [256] , [0,255])
            cv2.normalize(hist_item , hist_item , 0 , 255*0.9 , cv2.NORM_MINMAX)
            hist = np.int32(np.around(hist_item))
            pts = np.column_stack((bins , hist))
            cv2.polylines(h , [pts] , False , color)
        #hist = np.flipud(h)
        hist = h
        return hist
    else :
        print("flag must be 'GRAY' or 'BGR' or 'MULTICOLOR'")
        return False
            
img = cv2.imread("./ss.jpg")

h = histogram(img , flag="MULTICOLOR")
h = cv2PIL(h , flag="BGR2RGB")

ims = plt.imshow(h)
plt.show()

#cv2.imshow('colorhist',h)
#cv2.waitKey(0)