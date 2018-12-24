"""
美白模块
Whitening(img, x, y, w, h)
"""

import cv2
import numpy as np


def Whitening(img, x, y, w, h):
    """
    :param img:图片数组
    :param x:人脸框左上角x
    :param y:人脸框左上角y
    :param w:人脸框宽度
    :param h:人脸框高度
    :return:调整之后的图片数组
    """
    #import pdb
    #pdb.set_trace()
    img_clip = img[x:x+w,y:y+w,:]
    mask = skin_detect(img_clip)
    mask = mask.astype('uint8')
    img1 = cv2.bitwise_and(img_clip, mask)
    mask = cv2.bitwise_not(mask)
<<<<<<< HEAD
    img2 = cv2.bitwise_and(img_clip, mask)
=======
    img2 = cv2.bitwise_and(img, mask)
    # cv2.imshow("src1", img1)
>>>>>>> 70984457be8dc583b6adf432929c539f4ed0d412
    img3 = cv2.GaussianBlur(img1, (0, 0), 9)
    img1 = cv2.addWeighted(img1, 1.5, img3, -0.5, 0)

    img1 = cv2.bitwise_or(img1, img2)
<<<<<<< HEAD
    img[x:x + w, y:y + w, :] = img1
    add_temp = np.ones(img.shape, np.uint8) * 40
    img = cv2.add(img, add_temp)
    return img

=======
    img1 = cv2.add(img1, add_temp)
    return img1
    # cv2.imshow("src2", img2)
    # cv2.imshow("src3", img1)
    # cv2.waitKey(0)
>>>>>>> 70984457be8dc583b6adf432929c539f4ed0d412

def findBiggestContour(contours):
    #import pdb
    #pdb.set_trace()
    index = -1
    size = 0
    for i in range(len(contours)):
        if len(contours[i]) > size:
            size = len(contours[i])
            index = i
    return index


def skin_detect(img):
    #import pdb
    #pdb.set_trace()
    img_blur = cv2.blur(img, (3, 3))
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2YCrCb)
    cr = cv2.split(img_hsv)[1]
    _, cr = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cr_mask = np.zeros(img.shape)
    cr_mask[:, :, 0] = cr
    cr_mask[:, :, 1] = cr
    cr_mask[:, :, 2] = cr
    cr_mask = cr_mask.astype('uint8')
    mask = cv2.bitwise_and(img, cr_mask)
    return mask


if __name__ == '__main__':
    img = cv2.imread('./0171_01.jpg')
    img_ = whitening(img)
    cv2.imshow("img", img_)
    cv2.waitKey()
