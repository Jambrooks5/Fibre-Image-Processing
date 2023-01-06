import fibreAnalyser
import generalFunctions as gf

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import cv2
from datetime import datetime as dt

def main(img1, img2):
    centers = [getCenter(img1), getCenter(img2)]
    print("centers: ", centers)
    
    #Find the difference between the center of the two image, so they can be overlaid
    #xshift = centers[0][1] - centers[1][1]
    #yshift = centers[0][0] - centers[1][0]
    #print(xshift, yshift)
    
    plt.figure(dpi=300)
    t1 = dt.now()
    plt.imshow(img1, cmap='Blues', alpha=0.5, extent=[-centers[0][1], img1.shape[1]-centers[0][1], centers[0][0]-img1.shape[0], centers[0][0]])
    plt.imshow(img2, cmap='Reds', alpha=0.5, extent=[-centers[1][1], img2.shape[1]-centers[1][1], centers[1][0]-img2.shape[0], centers[1][0]])

    print(dt.now()-t1)

def getCenter(img):
    resizedChannel=gf.getSizeLimited(img,2**16)
    
    edgeChannel=fibreAnalyser.getAltSobelEdgeChnl(resizedChannel)
    
    #Find the amount of overlap when the image is reflected along different axes
    mirrorChannel=gf.getNormalised(fibreAnalyser.getMirrorSymmetryChannel(edgeChannel))*255
    #Finds the maximum point of the above to find the center of the fibre
    mirrorCenter=list(gf.getArrayMaxIndex(mirrorChannel))
    #The scale between the original image and the one used for mirroring
    scale = img.shape[0]/mirrorChannel.shape[0]
    center = [i*scale for i in mirrorCenter]
    
    print(center)
    return center


image1 = cv2.imread("KUV.jpg",0)
image2 = cv2.imread("KUVoffset.jpg",0)

main(image1, image2)