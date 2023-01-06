#https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('test2.png')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
#img=cv.drawKeypoints(gray,kp,img)
#cv.imwrite('sift_keypoints.jpg',img)


img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv.imwrite('sift_keypoints.jpg',img)
plt.figure(dpi=200)
plt.imshow(img)
plt.show()