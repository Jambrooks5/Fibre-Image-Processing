#https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('test2.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray,16,0.53,10)
corners = np.int0(corners)


for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),50,255,-1)

plt.figure(dpi=100)
plt.imshow(img),plt.show()