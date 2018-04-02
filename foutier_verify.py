#coding=utf-8

import sys
import time

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

import cv2

#设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'


def fft(img):
    '''对图像进行傅立叶变换，并返回换位后的频率矩阵'''
    assert img.ndim == 2, 'img should be gray.'

    rows, cols = img.shape[:2]

    # 计算最优尺寸
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)

    # 根据新尺寸，建立新变换图像
    nimg = np.zeros((nrows, ncols))
    nimg[:rows, :cols] = img

    # 傅立叶变换
    fft_mat = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)

    # 换位，低频部分移到中间，高频部分移到四周
    return np.fft.fftshift(fft_mat)


def fft_image(fft_mat):
    '''将频率矩阵转换为可视图像'''
    # log函数中加1，避免log(0)出现.
    log_mat = cv2.log(1 + cv2.magnitude(fft_mat[:, :, 0], fft_mat[:, :, 1]))

    # 标准化到0~255之间
    cv2.normalize(log_mat, log_mat, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(log_mat))


def ifft(fft_mat):
    '''傅立叶反变换，返回反变换图像'''
    # 反换位，低频部分移到四周，高频部分移到中间
    f_ishift_mat = np.fft.ifftshift(fft_mat)

    # 傅立叶反变换
    img_back = cv2.idft(f_ishift_mat)

    # 将复数转换为幅度, sqrt(re^2 + im^2)
    img_back = cv2.magnitude(*cv2.split(img_back))

    # 标准化到0~255之间
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(img_back))


if __name__ == '__main__':
    '''用bmp图测试，经过傅里叶变换和逆变换之后，图像有像素颜色值发生改变'''
    img = cv2.imread(sys.argv[1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure("傅里叶变换和反变换")
    plt.subplot(341), plt.imshow(gray,
                                 'gray'), plt.title('原图:' + str(gray.shape))
    ifft_mat = gray.copy()
    time0 = time.time()
    for i in range(0, 101):
        fft_mat = fft(ifft_mat)
        ifft_mat = ifft(fft_mat)
        if i % 10 == 0:
            plt.subplot(3, 4, i // 10 + 2), plt.imshow(
                ifft_mat,
                'gray'), plt.title('第' + str(i) + '次变换:' + str(fft_mat.shape))
    time1 = time.time()
    total = (time1 - time0)
    print("need time: %.3f s" % total)
    plt.show()
