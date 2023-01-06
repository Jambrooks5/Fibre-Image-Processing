import cv2 
import numpy as np

dataD = {}

dataD['img1'] = cv2.imread("KUV.jpg",0)
dataD['img2'] = cv2.imread("KUVoffset.jpg",0)

#overlay = cv2.addWeighted(dataD['img1'], 0.5, dataD['img2'], 0.7, 0)


img1 = dataD['img1'].copy()
img2 = dataD['img2'].copy()

xoff = yoff = 50
img1[yoff:yoff+img2.shape[0], xoff:xoff+img2.shape[1]] = img2
img1 = cv2.resize(img1, (int(img1.shape[1]/3), int(img1.shape[0]/3)))

cv2.imshow("Overlay", img1)

#####
'''
foreground = dataD['img1'].copy()
background = dataD['img1'].copy()

foreground_height = foreground.shape[0]
foreground_width = background.shape[1]
alpha =0.5

# do composite on the upper-left corner of the background image.
blended_portion = cv2.addWeighted(foreground,
            alpha,
            background[:foreground_height,:foreground_width],
            1 - alpha,
            0,
            background)
background[:foreground_height,:foreground_width] = blended_portion

cv2.imshow("Overlay", background)
#####
'''

'''
#######
anchor_y = 10
anchor_x = 10

background_height = background.shape[0]
background_width = background.shape[1]
foreground_height = foreground.shape[0]
foreground_width = foreground.shape[1]
if foreground_height+anchor_y > background_height or foreground_width+anchor_x > background_width:
    raise ValueError("The foreground image exceeds the background boundaries at this location")

alpha =0.5

# do composite at specified location
start_y = anchor_y
start_x = anchor_x
end_y = anchor_y+foreground_height
end_x = anchor_x+foreground_width
blended_portion = cv.addWeighted(foreground,
            alpha,
            background[start_y:end_y, start_x:end_x,:],
            1 - alpha,
            0,
            background)
background[start_y:end_y, start_x:end_x,:] = blended_portion
cv.imshow('composited image', background)
#######

'''
#int(img1.shape[0]/2), int(img1.shape[1]/2

cv2.waitKey(0)
cv2.destroyAllWindows()