import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.Image

from skimage import data,color,measure
from PIL import ImageTk
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from tkinter import *


def get_masked_image(_image):
    image_array = np.array(_image)
    outp_image_array = rgb2gray( image_array[...,0:3] )# converts to greyscale
    
    _mask_value = get_array_average(outp_image_array)########################
    
    mask_array = outp_image_array <= _mask_value
    outp_image_array[mask_array] = 0# sets values <= _mask_value to 0
    mask_array = outp_image_array > _mask_value
    outp_image_array[mask_array] = 255# sets values > _mask_value to 255
    
    outp_image = PIL.Image.fromarray(outp_image_array)# converts to final output image format
    return outp_image

def get_array_average(_array): 
    return np.sum(_array) / _array.size

image_names = ["untapered.png", "uncleavedneedleburner.png"]

for image_name in image_names:
    image = PIL.Image.open(image_name)
    
    masked_image=get_masked_image(image)
    masked_image.show()

    
'''
#convert image to array
image=np.array(image)
#print(image)
#turn image only full black or white
image = rgb2gray(image[...,0:3])
#print (image)

print(np.max(image))
#print("Mask start")
mask = image < np.max(image)/4
image[mask] = 0

mask = image > np.max(image)/4

image[mask] = 255

Image.fromarray(image).show()
'''
#print("Mask stop, edges start")

'''
edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)
print("Edges stop, Hugh radius start")

#for i in edges: print( set([x==True for x in i]) )

# Detect two radii
hough_radii = np.arange(50, 1200, 100)

print("Hugh res start")
hough_res = hough_circle(edges, hough_radii)
#print(hough_res)

print("accums start")
# Select the most prominent 3 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=2)

# Draw them
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(100, 40))
image = color.gray2rgb(image)
#x=0###
#ax.imshow(image)
print("For loop start")
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=image.shape)
    ##if x==0: print(circy, circx)###
    #x=1###
    #ax.plot(circx,circy,linestyle='',markersize=101, color='r')
    image[circy, circx] = (1, 0, 0)

ax.imshow(image)
plt.show()

'''
#Image.fromarray(image).show()

'''
plt.figure(figsize=(16, 16))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()


# Find contours at a constant value of 0.8
contours = measure.find_contours(img, 0.3)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(img, cmap=plt.cm.gray)



#for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    #print (contour[:,1], contour[:,0], "\n" )
    #print(len(contour))
    #print("Fibre diam: ", max(contour[:,0])- min(contour[:,0]))
    
    #print(contour)

'''

'''
plt.figure(figsize=(16, 16))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
'''






