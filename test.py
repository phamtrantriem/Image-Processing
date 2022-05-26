from shutil import copyfile

import cv2
import numpy as np


def test():
    value = 40
    copyfile("static/img/img_normal.jpg", "static/img/img_now.jpg")
    image = cv2.imread("static/img/img_now.jpg")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite("static/img/img_now.jpg", img)

test()
