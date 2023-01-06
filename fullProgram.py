import CallableMasker
import callableCircleFinder
import simonsMagic
import fromScratch
import CallableGUICapFinder
import manualCenterFinder

import numpy as np
import cv2
import tkinter as tk
import PIL

from datetime import datetime as dt


def chooseMask():
    choice = ""
    originalImage = cv2.imread("KUV.jpg",0)
    
    while (choice!="b" and choice!="s"):
        choice = input("Type 'b' for basic masking only, or 's' for applying sobel operator to image before basic masking:")
        
        if (choice=="b"):
            maskedImage = (CallableMasker.main(originalImage))
        elif (choice=="s"):
            sobelImage = simonsMagic.main(originalImage)
            sobelImage = np.array(sobelImage.astype(np.uint8))
            maskedImage = (CallableMasker.main(sobelImage))
        else:
            print("Invalid choice, enter b or s, lower case")

    return maskedImage

def getCircleData(maskedImage):
    centerChoice = ""

    while (centerChoice!="c" and centerChoice!="m"):
        centerChoice = input("Type 'c' to use the circle finding program to find either capillaries or the core, or 'm' to manually click on the center of the core:")
        
        if (centerChoice=="c"):
            circleData = callableCircleFinder.main(maskedImage, True)
            centerX, centerY = fromScratch.capCenterFinder(circleData)
        elif (centerChoice=="m"):
            circleData = "null"
            centerX, centerY = manualCenterFinder.main(maskedImage)
        else:
            print("Invalid choice, enter c or m, lower case")

    return circleData, centerX, centerY


def main():
    maskedImage = chooseMask()
    
    circleData, centerX, centerY = getCircleData(maskedImage)




main()