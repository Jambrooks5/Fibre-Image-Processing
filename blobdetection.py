# Standard imports
import cv2
import PIL
import numpy as np;

# Read image
im = cv2.imread("blobtest.png", cv2.IMREAD_GRAYSCALE )
#im = cv2.bitwise_not(im)
# Set up the detector with default parameters.
#PIL.Image.fromarray(im).show()
detector = cv2.SimpleBlobDetector()

# Detect blobs.

keypoints = detector.detect(im)



# Draw detected blobs as red circles.

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

 

# Show keypoints

cv2.imshow("Keypoints", im_with_keypoints)

cv2.waitKey(0)
