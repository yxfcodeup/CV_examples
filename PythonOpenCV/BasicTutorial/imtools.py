import os
import sys
#External Libs
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

#------------------------------------------------------------------------------
# 直方图
# @param image 三通道图片
# @param flag BGR2RGB/RGB2BGR
# @return image
# NOTICE:
#------------------------------------------------------------------------------
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


def erodeDilate(cv_img) :
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (9,9))  #opencv定义的结构元素
    #kernel = np.uint8(np.ones((9,9)))  #numpy定义的结构函数，与上等同
    eroded = cv2.erode(cv_img , kernel)    #腐蚀图像
    dilated = cv2.dilate(cv_img , kernel)  #膨胀图像
    return (eroede , dilated)


#开运算：去除较小的明亮区域
#闭运算：消除低亮度值的孤立点
def openCloseOP(cv_img) :
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (9,9))
    opened = cv2.morphologyEx(cv_img , cv2.MORPH_OPEN , kernel)
    #closed = cv2.morphologyEx(cv_img , cv2.MORPH_CLOSE , kernel)
    closed = cv2.morphologyEx(opened , cv2.MORPH_CLOSE , kernel)
    return (opened , closed)


"""
形态学检测边缘的原理很简单，在膨胀时，图像中的物体会想周围“扩张”；腐蚀时，图像中的物体会“收缩”。比较这两幅图像，由于其变化的区域只发生在边缘。所以这时将两幅图像相减，得到的就是图像中物体的边缘。
"""
def checkEdge(cv_img) :
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (3,3))
    dilate = cv2.dilate(cv_img , kernel)
    erode = cv2.erode(cv_img , kernel)
    res = cv2.absdiff(dilate , erode)   #将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
    ret_val , res = cv2.threshold(res , 40 , 255 , cv2.THRESH_BINARY)   #上面得到的结果是灰度图，将其二值化以便更清楚的观察结果
    res = cv2.bitwise_not(res)  #反色，即对二值图每个像素取反
    return res


"""
与边缘检测不同，拐角的检测的过程稍稍有些复杂。但原理相同，所不同的是先用十字形的结构元素膨胀像素，这种情况下只会在边缘处“扩张”，角点不发生变化。接着用菱形的结构元素腐蚀原图像，导致只有在拐角处才会“收缩”，而直线边缘都未发生变化。
第二步是用X形膨胀原图像，角点膨胀的比边要多。这样第二次用方块腐蚀时，角点恢复原状，而边要腐蚀的更多。所以当两幅图像相减时，只保留了拐角处。
"""
def checkCorner(cv_img) :
    #构造5×5的结构元素，分别为十字形、菱形、方形和X型
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS , (5,5))
    diamond = cv2.getStructuringElement(cv2.MORPH_RECT , (5,5))
    diamond[0, 0] = 0
    diamond[0, 1] = 0
    diamond[1, 0] = 0
    diamond[4, 4] = 0
    diamond[4, 3] = 0
    diamond[3, 4] = 0
    diamond[4, 0] = 0
    diamond[4, 1] = 0
    diamond[3, 0] = 0
    diamond[0, 3] = 0
    diamond[0, 4] = 0
    diamond[1, 4] = 0
    square = cv2.getStructuringElement(cv2.MORPH_RECT , (5,5))
    x_shape = cv2.getStructuringElement(cv2.MORPH_CROSS , (5,5))

    res1 = cv2.dilate(cv_img , cross)
    res1 = cv2.erode(res1 , diamond)
    res2 = cv2.dilate(cv_img , x_shape)
    res2 = cv2.erode(res2 , square)
    res = cv2.absdiff(res2 , res1)
    ret_val , result = cv2.threshold(result , 40 , 255 , cv2.THRESH_BINARY)

    for j in range(result.size) :
        y = j / result.shape[0]
        x = j % result.shape[0]
        if 255 == result[x , y] :
            cv2.circle(cv_image , (y,x))


if "__main__" == __name__ :
    #testHistogram()
    img = cv2.imread("./saber_cos.jpg" , 0)
    a = checkEdge(img)
    cv2.imshow("a" , a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
