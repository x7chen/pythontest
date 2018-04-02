# -*- coding: UTF-8 -*-
import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

import cv2 as cv

#设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
#读取字体目录
print(matplotlib.matplotlib_fname())
#读取图像，支持 bmp、jpg、png、tiff 等常用格式
img = cv.imread(sys.argv[1])
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_r = np.zeros(img.shape, img.dtype)
img_r[:, :, 0] = img[:, :, 0]
img_g = np.zeros(img.shape, img.dtype)
img_g[:, :, 1] = img[:, :, 1]
img_b = np.zeros(img.shape, img.dtype)
img_b[:, :, 2] = img[:, :, 2]

(R, G, B) = cv.split(img)

print(B.shape)
#cv.imshow("B",B)
#img_b = cv.cvtColor(img_b,cv.COLOR_RGB2GRAY)
#创建窗口并显示图像
plt.figure("颜色分量")
plt.subplot(221), plt.imshow(img), plt.title('原图')
plt.subplot(222), plt.imshow(R, 'gray'), plt.title('红色分量')
plt.subplot(223), plt.imshow(G, 'gray'), plt.title('绿色分量')
plt.subplot(224), plt.imshow(B, 'gray'), plt.title('蓝色分量')
plt.show()
# cv.namedWindow("Image")
# cv.imshow("Image",img)
cv.waitKey(0)
#释放窗口
cv.destroyAllWindows()
