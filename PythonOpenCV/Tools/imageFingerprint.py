import os
import sys
from functools import reduce
#External Libs
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from basics import cv2PIL


"""
aHash：速度比较快，但是常常不太精确。
pHash：精确度比较高，但是速度方面较差一些。
dHash：精确度较高，且速度也非常快。常用此方法用来做图片判重
"""


#------------------------------------------------------------------------------
# 计算平局hash值，通过比较
# @param cv_img 通过cv2.imread读取的图片，未经过灰度处理
# @return hash列表
# NOTICE:
#------------------------------------------------------------------------------
def getImageHash(cv_img) :
    dim = cv_img.ndim
    avg = np.mean(cv_img)
    res_hash = list()
    if 2 == dim :
        for h in range(cv_img.shape[0]) :
            for w in range(cv_img.shape[1]) :
                if cv_img[h , w] > avg :
                    res_hash.append(1)
                else :
                    res_hash.append(0)
    elif 3 == dim :
        chn = cv_img.shape[2]
        for h in range(cv_img.shape[0]) :
            for w in range(cv_img.shape[1]) :
                for n in range(chn) :
                    if cv_img[h , w , n] > avg :
                        res_hash.append(1)
                    else :
                        res_hash.append(0)
    else :
        print("ERROR: shape of cv_img must be 2 or 3!")
        return res_hash
    return res_hash


#------------------------------------------------------------------------------
# 求取两个等长字符串(二进制形式或者list形式)之间的hamming距离
# @param ojb1 一段二进制字符串或者由0、1组成的list
# @param ojb2 一段二进制字符串或者由0、1组成的list，与obj1等长
# @return hamming距离值
# NOTICE:
#------------------------------------------------------------------------------
def hammingDistance(obj1 , obj2) :
    if len(obj1) != len(obj2) :
        print("ERROR: In hamming distance,obj1 and obj2 must be same length!")
        return False
    if (not isinstance(obj1 , str)) and (not isinstance(obj2 , str)) and (not isinstance(obj1 , list)) and (not isinstance(obj2 , list)) :
        print("ERROR: In hamming distance,ojb1 and obj2 must be binary format or a list which is made up of 0 and 1!")
        print("obj1: " , obj1)
        print("obj2: " , obj2)
        return False
    dist = 0
    for i in range(len(obj1)) :
        if obj1[i] != obj2[i] :
            dist += 1
    return dist

#------------------------------------------------------------------------------
# 平均hash法(aHash)计算两图片相似度，主要用于灰度缩略图的相似度比较，精度不够，适合搜索缩略图
# @param cv_img1 通过cv2.imread读取的图片，未经过灰度处理
# @param cv_img2 通过cv2.imread读取的图片，未经过灰度处理
# @param size 缩放大小，为了保留结构去掉细节，去除大小、横纵比的差异，默认把图片统一缩放到8*8，共64个像素的图片
# @return hamming距离值，值越大相似度越小
# NOTICE:
#------------------------------------------------------------------------------
def aHash(cv_img1 , cv_img2 , size=(8,8)) :
    img1 = cv2.resize(cv_img1 , size)
    gray1 = cv2.cvtColor(img1 , cv2.COLOR_BGR2GRAY)
    #avg1 = gray1.sum() / reduct(lambda x,y:x*y , list(gray1.shape))
    hash1 = getImageHash(gray1)
    img2 = cv2.resize(cv_img2 , size)
    gray2 = cv2.cvtColor(img2 , cv2.COLOR_BGR2GRAY)
    #avg2 = gray2.sum() / reduct(lambda x,y:x*y , list(gray2.shape))
    hash2 = getImageHash(gray2)
    hdist = hammingDistance(hash1 , hash2)
    return hdist


#------------------------------------------------------------------------------
# 感知hash法(aHash)计算两图片相似度，采用DCT(离散余弦变换)来降低频率的方法，精度大于aHash
# @param cv_img1 通过cv2.imread读取的图片，未经过灰度处理
# @param cv_img2 通过cv2.imread读取的图片，未经过灰度处理
# @param size 缩放大小，默认把图片统一缩放到32*32，便于DCT计算
# @return hamming距离值，值越大相似度越小
# NOTICE:
#------------------------------------------------------------------------------
def pHash(cv_img1 , cv_img2 , size=(32,32)) :
    img1 = cv2.resize(cv_img1 , size)
    img2 = cv2.resize(cv_img2 , size)
    gray1 = cv2.cvtColor(img1 , cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2 , cv2.COLOR_BGR2GRAY)
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    #取左上角8*8，这些代表图片的最低频率
    dct1_roi = dct1[:8 , :8]
    dct2_roi = dct2[:8 , :8]
    hash1 = getImageHash(dct1_roi)
    hash2 = getImageHash(dct2_roi)
    hdist = hammingDistance(hash1 , hash2)
    return hdist


#------------------------------------------------------------------------------
# 差异值hash法(dHash)计算两图片相似度
# @param cv_img1 通过cv2.imread读取的图片，未经过灰度处理
# @param cv_img2 通过cv2.imread读取的图片，未经过灰度处理
# @param rows 图片缩放后行数，列数通过行数计算，默认为8
# @return hamming距离值，值越大相似度越小
# NOTICE: 一般来说，汉明距离小于5，基本就是同一张图片
#------------------------------------------------------------------------------
def dHash(cv_img1 , cv_img2 , rows=8) :
    cols = rows + 1
    # 1.缩放图片
    img1 = cv2.resize(cv_img1 , (cols , rows))
    img2 = cv2.resize(cv_img2 , (cols , rows))
    # 2.灰度化
    gray1 = cv2.cvtColor(img1 , cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2 , cv2.COLOR_BGR2GRAY)
    # 3.差异计算，比较相邻像素，并构成二进制hash字符串
    hash1 = str()
    hash2 = str()
    for h in range(rows) :
        for w in range(rows) :
            hash1 += "1" if gray1[h , w] > gray1[h , w+1] else "0"
            hash2 += "1" if gray2[h , w] > gray2[h , w+1] else "0"
    hdist = hammingDistance(hash1 , hash2)
    return hdist


if "__main__" == __name__ :
    img1 = cv2.imread("./t1.jpg")
    img2 = cv2.imread("./t2.jpg")
    a = aHash(img1 , img2)
    b = pHash(img1 , img2)
    c = dHash(img1 , img2)
    print(a)
    print(b)
    print(c)
