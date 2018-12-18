"""
双边滤波的实现
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

img_path = '/lena.jpg'

def get_weight(i, j , k, l, img, coef_d, coef_r):
    return math.exp(
        -((pow(i-k,2)+pow(j-l,2))/(2*coef_d*coef_d))
        -(pow(img[i][j]-img[k][l],2)/(2*coef_r*coef_r))
        )

def Bilateral_filter(img, x, y, w, h):
    """
        双边滤波器的手动实现
        :param img_path: 图片路径
        :param x: 滤波区域左上角x
        :param y: 滤波区域左上角y
        :param w: 滤波区域x宽度
        :param h: 滤波区域y高度
        :return:
        """
    img_clip = img[x:x+w,y:y+w,:]
    img_cliped = cv2.bilateralFilter(src=img_clip, d=0, sigmaColor=100, sigmaSpace=15)
    cv2.namedWindow('before', 0)
    cv2.resizeWindow('before', 300, 400)
    cv2.imshow("before", img_clip)
    cv2.namedWindow('demo',0)
    cv2.resizeWindow('demo',300,400)
    cv2.imshow("demo", img_cliped)
    cv2.waitKey(0)


def remove_beverage(img):
    return cv2.bilateralFilter(src=img, d=0, sigmaColor=100, sigmaSpace=15)


def Bilateral_filter_old(img, x, y, w, h, size):
    """
    双边滤波器的手动实现
    :param img_path: 图片路径
    :param x: 滤波区域左上角x
    :param y: 滤波区域左上角y
    :param w: 滤波区域x宽度
    :param h: 滤波区域y高度
    :param size: 滤波器卷积核尺寸
    :return:
    """
    #import pdb
    #pdb.set_trace()
    b, g, r = cv2.split(img)
    b_new = np.zeros(b.shape)
    g_new = np.zeros(g.shape)
    r_new = np.zeros(r.shape)
    for i in range(w):
        for j in range(h):
            #print('({},{})'.format(i, j))
            value = 0
            weight = 0
            if x+i < size/2 or x+i > img.shape[0]-size/2 or y+j <size/2 or y+j > img.shape[1]-size/2:
                b_new[x+i][y+j] = b[x+i][y+j]
            else:
                for p in range(int(-size/2), int(size/2)):
                    for q in range(int(-size/2), int(size/2)):
                        value += int(b[x+i+p][y+j+q]*get_weight(x+i, y+j, x+i+p, y+j+q, b, 3, 3))
                        weight += int(get_weight(x+i, y+j, x+i+p, y+j+q, b, 3, 3))
                b_new[x+i][y+j] = value/weight
    print('b_over')
    for i in range(w):
        for j in range(h):
            #print('({},{})'.format(i, j))
            value = 0
            weight = 0
            if x+i < size/2 or x+i > img.shape[0]-size/2 or y+j <size/2 or y+j > img.shape[1]-size/2:
                g_new[x+i][y+j] = g[x+i][y+j]
            else:
                for p in range(int(-size/2), int(size/2)):
                    for q in range(int(-size/2), int(size/2)):
                        value += int(g[x+i+p][y+j+q]*get_weight(x+i, y+j, x+i+p, y+j+q, g, 3, 3))
                        weight += int(get_weight(x+i, y+j, x+i+p, y+j+q, g, 3, 3))
                g_new[x+i][y+j] = value/weight
    print('g_over')
    for i in range(w):
        for j in range(h):
            value = 0
            weight = 0
            if x+i < size/2 or x+i > img.shape[0]-size/2 or y+j <size/2 or y+j > img.shape[1]-size/2:
                r_new[x+i][y+j] = r[x+i][y+j]
            else:
                for p in range(int(-size/2), int(size/2)):
                    for q in range(int(-size/2), int(size/2)):
                        value += int(r[x+i+p][y+j+q]*get_weight(x+i, y+j, x+i+p, y+j+q, r, 3, 3))
                        weight += int(get_weight(x+i, y+j, x+i+p, y+j+q, r, 3, 3))
                r_new[x+i][y+j] = value/weight
    print('r_over')
    img_new = cv2.merge([r_new, g_new, b_new])
    plt.imshow(img_new)
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('./lena.jpg')
    Bilateral_filter(img, 0, 0, img.shape[0], img.shape[1])
