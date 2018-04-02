# -*- coding:utf-8 -*-
__author__ = 'Sean'

import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# surf.hessianThreshold=3000
# surf = cv2.SURF(3000)
surf = cv2.xfeatures2d.SURF_create(3000)
kp, res = surf.detectAndCompute(gray, None)
print(res.shape)

img = cv2.drawKeypoints(img, kp, None, (255, 0, 255), 4)
print(len(kp))

cv2.namedWindow("SURF")
cv2.imshow("SURF", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
