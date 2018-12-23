"""
美白模块
"""

import cv2
import numpy as np


def whitening(img, mask):
    #import pdb
    #pdb.set_trace()
    mask = mask.astype('uint8')
    img1 = cv2.bitwise_and(img, mask)
    mask = cv2.bitwise_not(mask)
    img2 = cv2.bitwise_and(img, mask)
    cv2.imshow("src1", img1)
    img3 = cv2.GaussianBlur(img1, (0, 0), 9)
    img1 = cv2.addWeighted(img1, 1.5, img3, -0.5, 0)
    add_temp = np.ones(img1.shape, np.uint8)*40
    img1 = cv2.bitwise_or(img1, img2)
    img1 = cv2.add(img1, add_temp)
    cv2.imshow("src2", img2)
    cv2.imshow("src3", img1)
    cv2.waitKey(0)

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
    '''
    cv2.imshow("src", mask)
    cv2.waitKey(0)
    lower = np.array([0, 133, 20])
    upper = np.array([255, 180, 127])
    mask = cv2.inRange(img_hsv, lower, upper)
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index = findBiggestContour(contours)
    output = np.zeros(img.shape)
    output = cv2.drawContours(output, contours, index, (255,255,255), -1, hierarchy=hierarchy)
    return output
    '''

    #cv2.imshow("src", img)
    #cv2.waitKey(0)




if __name__ == '__main__':
    img = cv2.imread('./0171_01.jpg')
    mask = skin_detect(img)
    whitening(img, mask)