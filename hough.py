import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #灰度图像
edges = cv2.Canny(gray, 50, 200)
drawing = np.zeros(img.shape[:], dtype=np.uint8)
#(函数参数3和参数4) 通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
#118 --是经过某一点曲线的数量的阈值
lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)  #这里对最后一个参数使用了经验型的值
result = img.copy()
for line in lines: 
    rho, theta = line[0] 
    a = np.cos(theta) 
    b = np.sin(theta) 
    x0 = a * rho 
    y0 = b * rho 
    x1 = int(x0 + 1000 * (-b)) 
    y1 = int(y0 + 1000 * (a)) 
    x2 = int(x0 - 1000 * (-b)) 
    y2 = int(y0 - 1000 * (a)) 
    cv2.line(drawing, (x1, y1), (x2, y2), (0, 0, 255))

cv2.imshow('Canny', edges)
cv2.imshow('Result', result)
cv2.imshow('Hough', drawing)
cv2.waitKey(0)
cv2.destroyAllWindows()
