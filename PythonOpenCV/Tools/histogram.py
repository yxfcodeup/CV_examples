import os
import sys
#External Libs
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from basics import cv2PIL

#------------------------------------------------------------------------------
# 直方图
# @param cv_img 通过cv2.imread读取的图片，未经过灰度处理
# @param flag 三种模式：GRAY为处理成灰度直方图；BGR为处理成BGR三种颜色单独的直方图；MULTICOLOR为处理成三种颜色为同一张图片的直方图
# @param is_img 是否是生成bgr三通道图片形式
# @return bgr三通道图片，或者单直方图二维数组
# NOTICE:
#------------------------------------------------------------------------------
def histogram(cv_img , flag="GRAY" , is_img=False) :
    def calcAndDrawHist(img , color) :
        hist = cv2.calcHist([img] ,  
                [0] ,           #使用的通道  
                None ,          #没有使用mask  
                [256] ,         #HistSize  
                [0.0,255.0])    #直方图柱的范围
        min_val , max_val , min_loc, max_loc = cv2.minMaxLoc(hist)
        hist_img = np.zeros([256,256,3] , np.uint8)
        hpt = 256    #直方图最大高度
        for h in range(256) :
            intensity = int(hist[h] * hpt / max_val)
            cv2.line(hist_img , (h,256) , (h,256-intensity) , color)
        return hist_img

    if "GRAY" == flag :
        hist = None
        gray_img = cv2.cvtColor(cv_img , cv2.COLOR_BGR2GRAY)
        if is_img :
            hist = calcAndDrawHist(gray_img , [128,128,128])
        else :
            hist = cv2.calcHist([gray_img] ,  
                    [0] ,           #使用的通道  
                    None ,          #没有使用mask  
                    [256] ,         #HistSize  
                    [0.0,255.0])    #直方图柱的范围
        return hist
    elif "BGR" == flag :
        hist_b = hist_g = hist_r = None
        b , g , r = cv2.split(cv_img)
        if is_img :
            hist_b = calcAndDrawHist(b , [255,0,0])
            hist_g = calcAndDrawHist(g , [0,255,0])
            hist_r = calcAndDrawHist(r , [0,0,255])
        else :
            hist_b = cv2.calcHist([b] , [0] , None , [256] , [0.0,255.0])
            hist_g = cv2.calcHist([g] , [0] , None , [256] , [0.0,255.0])
            hist_r = cv2.calcHist([r] , [0] , None , [256] , [0.0,255.0])
        return (hist_b , hist_g , hist_r)
    elif "MULTICOLOR" == flag :
        h = np.zeros([256,256,3] , np.uint8) #创建用于绘制直方图的全0图像
        bins = np.arange(256).reshape(256,1)    #直方图中各bin的顶点位置
        colors = [(255,0,0) , (0,255,0) , (0,0,255) ]   #BGR三种颜色
        for channel , color in enumerate(colors) :
            hist_item = cv2.calcHist([cv_img] , [channel] , None , [256] , [0.0,255.0])
            cv2.normalize(hist_item , hist_item , 0 , 256 , cv2.NORM_MINMAX)    #归一化处理，将直方图的范围限定在0-256之间
            hist = np.int32(np.around(hist_item))       #将生成的原始直方图中的每个元素四舍六入五凑偶取整（cv2.calcHist函数得到的是float32类型的数组）
            pts = np.column_stack((bins , hist))        #将直方图中每个bin的值转成相应的坐标
            cv2.polylines(h , [pts] , False , color)    #根据这些点绘制出折线，第三个False参数指出这个折线不需要闭合。第四个参数指定了折线的颜色。
        hist = np.flipud(h)     #反转绘制好的直方图，因为绘制时，[0,0]在图像的左上角。
        return hist
    else :
        print("flag must be 'GRAY' or 'BGR' or 'MULTICOLOR'")
        return False

def splitImage(img , part_size=(0,0)) :
    if (0,0) == part_size :
        return img
    ph , pw = part_size
    h , w = (img.shape)[:2]
    if w % pw == h % ph == 0 :
        print(w , pw , h , ph)
        print("ERROR: split imgae error!")
        return img

    all_sub_imgs = list()
    for i in range(0 , h , ph) :
        for j in range(0 , w , pw) :
            sub_img = img[i:i+ph , j:j+pw]
            all_sub_imgs.append(sub_img)
    return all_sub_imgs


#------------------------------------------------------------------------------
# 使用直方图计算两图片的相似度
# @param cv_img1 通过cv2.imread读取的图片，未经过灰度处理
# @param cv_img2 通过cv2.imread读取的图片，未经过灰度处理
# @param size 将图片重新制定大小，默认为不指定，(height , length)
# @param part_size
# @return 相似度，1.0为完全相似，0为完全不相似
# NOTICE:
#------------------------------------------------------------------------------
def calcSimilarHist(cv_img1 , cv_img2 , size=(0,0)) :
    if (0,0) != size :
        cv_img1 = cv2.resize(cv_img1 , size)
        cv_img2 = cv2.resize(cv_img2 , size)
    hb1 , hg1 , hr1 = histogram(cv_img1 , flag="BGR")
    hb2 , hg2 , hr2 = histogram(cv_img2 , flag="BGR")
    x = (np.concatenate((hb1 , hg1 , hr1) , axis=0))[: , 0]
    y = (np.concatenate((hb2 , hg2 , hr2) , axis=0))[: , 0]
    if len(x) != len(y) :
        print(len(x) , " != " , len(y))
        return False
    data = []
    for i in range(len(x)) :
        if x[i] != y[i] :
            r = 1 - abs(x[i] - y[i]) / max(x[i] , y[i])
            data.append(r)
        else :
            data.append(1)
    res = sum(data) / len(x)
    return res


#------------------------------------------------------------------------------
# 使用POOL形式的直方图计算两图片的相似度，即将图片分割成小图片形式计算
# @param cv_img1 通过cv2.imread读取的图片，未经过灰度处理
# @param cv_img2 通过cv2.imread读取的图片，未经过灰度处理
# @param size 将图片重新制定大小，默认为不指定，(height , length)
# @param part_size 子图片大小，默认为不分割，(height , length)
# @param is_normal 是否以较小图片置为同一宽高，默认为是。
# @return 相似度，1.0为完全相似，0为完全不相似
# NOTICE:
#------------------------------------------------------------------------------
def calcSimilarHistSplit(cv_img1 , cv_img2 , size=(0,0) , part_size=(0,0) , is_normal=True) :
    res = 0
    minh = cv_img1.shape[0]
    minl = cv_img1.shape[1]
    if (0,0) == part_size :
        res = calcSimilarHist(cv_img1 , cv_img2 , size)
    else :
        if cv_img1.shape != cv_img2.shape :
            if cv_img1.shape[2] != cv_img2.shape[2] :
                print("WARNING: Channels of two images are not equal!")
                return 0
            if cv_img1.shape[0] * cv_img2.shape[1] != cv_img1.shape[1] * cv_img2.shape[0] :
                print("WARNING: Height and length is not proportionable!")
                return 0
            else :
                minh = min(cv_img1.shape[0] , cv_img2.shape[0])
                minl = min(cv_img1.shape[1] , cv_img2.shape[1])
                cv_img1 = cv2.resize(cv_img1 , (minh , minl))
                cv_img2 = cv2.resize(cv_img2 , (minh , minl))
        sub1 = splitImage(cv_img1 , part_size)
        sub2 = splitImage(cv_img2 , part_size)
        sub_num = 0
        for im1 , im2 in zip(sub1 , sub2) :
            sub_num += 1
            res += calcSimilarHist(im1 , im2 , size)
        x = minl / part_size[1]
        y = minh / part_size[0]
        res = res / sub_num
    return res


def testHistogram() :
    img = cv2.imread("./saber_cos.jpg")
    #img = cv2.imread("./saber_cos.jpg" , 0)    #直接读为灰度图像

    #hist_cv = cv2.calcHist([img] , [0] , None , [256] , [0,256])
    #hist_np , bins = np.histogram(img.ravel() , 256 , [0,256])
    #hist_np2 = np.bincount(img.ravel() , minlength=256)

    #hist = histogram(img , flag="GRAY")
    #hb , hg , hr = histogram(img , flag="BGR" , is_img=True)
    hist_m = histogram(img , flag="MULTICOLOR")
    img = cv2PIL(img , flag="BGR2RGB")
    #hb = cv2PIL(hb , flag="BGR2RGB")
    #hg = cv2PIL(hg , flag="BGR2RGB")
    #hr = cv2PIL(hr , flag="BGR2RGB")
    hist_m = cv2PIL(hist_m , flag="BGR2RGB")

    #plt.subplot(221)
    #plt.imshow(img)

    #plt.subplot(222)
    #plt.plot(hist_cv)
    #plt.subplot(223)
    #plt.plot(hist_np)
    #plt.subplot(224)
    #plt.plot(hist_np2)
    #plt.subplot(222)
    #plt.plot(hist)

    #plt.subplot(222)
    #plt.plot(hb)
    #plt.subplot(223)
    #plt.plot(hg)
    #plt.subplot(224)
    #plt.plot(hr)

    #plt.subplot(222)
    #plt.imshow(hb)
    #plt.subplot(223)
    #plt.imshow(hg)
    #plt.subplot(224)
    #plt.imshow(hr)

    #plt.subplot(222)
    plt.imshow(hist_m)
    plt.show()


    #cv2.imshow("img" , img)
    #cv2.imshow("hist_b" , hb)
    #cv2.imshow("hist_g" , hg)
    #cv2.imshow("hist_r" , hr)
    #cv2.waitKey(0)

if "__main__" == __name__ :
    img1 = cv2.imread("./t1.jpg")
    img2 = cv2.imread("./t2.jpg")
    a = calcSimilarHistSplit(img1 , img2 , (0,0) , (50,50))
    b = calcSimilarHistSplit(img1 , img2 , (0,0) , (0,0))
    i = 1
    while True :
        if i*10 > min(min(img1.shape[0] , img1.shape[1]) , min(img2.shape[0] , img2.shape[1])) :
            break
        a = calcSimilarHistSplit(img1 , img2 , (0,0) , (i*10+1 , i*10+1))
        i += 1
        print(a , "\t" , b , "\t" , a-b)
    #subs1 = splitImage(img1 , (100 , 100))
    #subs2 = splitImage(img2 , (100 , 100))
    #cv2.imshow("img2" , img2)
    #cv2.imshow("img1" , img1)
    #cv2.imshow("subs1[0]" , subs1[0])
    #cv2.imshow("subs2[0]" , subs2[0])
    #cv2.waitKey(0)
    #h1 = histogram(img1 , flag="MULTICOLOR" , is_img=True)
    #h2 = histogram(img2 , flag="MULTICOLOR" , is_img=True)
    #plt.subplot(221)
    #plt.imshow(h1)
    #plt.subplot(222)
    #plt.imshow(h2)
    #plt.show()
