# -*- coding:utf-8 -*-
__author__ = 'Sean'

import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# kp 是所有128特征描述子的集合
kp = sift.detect(gray, None)
print(len(kp))

# 找到后可以计算关键点的描述符
Kp, res = sift.compute(gray, kp)
print(Kp)  # 特征点的描述符
print(res)  # 是特征点个数*128维的矩阵

# 还可以用下面的函数直接检测并返回特征描述符
kp2, res1 = sift.detectAndCompute(gray, None)
print("******************************")
print(res1)

img = cv2.drawKeypoints(img, kp, img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("sift", img)
cv2.waitKey(0)
cv2.destroyAllWindows()