#-*- coding: utf-8 -*-

__author__ = 'Sean'

import cv2
import os
import time
import sys
if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    time0 = time.time()
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    time1 = time.time()
    total = (time1 - time0)
    print ("otsu need time: %.3f s" % total)

    cv2.imshow("src", img)
    cv2.imshow("gray", gray)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
