# -*- coding:utf-8 -*-
__author__ = 'Sean'

import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#fast = cv2.FastFeatureDetector(threshold=15)
fast = cv2.FastFeatureDetector_create(15)
kp = fast.detect(gray, None)

img = cv2.drawKeypoints(img, kp, img, color=(255, 0, 0))

# print all default parms
#print("Threshold: ", fast.getInt('threshold'))
#print("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
#print "neighborhood: ", fast.getInt('type')
#print("Total Keypoints with nonmaxSuppression: ", len(kp))

# disable nonmaxSuppression
#fast.setBool('nonmaxSuppression', 0)
kp = fast.detect(gray, None)

print("Total Keypoints without nonmaxSuppression: ", len(kp))
img2=img
cv2.drawKeypoints(img, kp, img2, color=(255, 0, 0))

res = np.hstack((img, img2))

cv2.namedWindow("Fast")
cv2.imshow("Fast", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
