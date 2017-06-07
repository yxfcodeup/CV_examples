import os
import sys
#External Libs
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

#------------------------------------------------------------------------------
# BGR <--> RGB
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
