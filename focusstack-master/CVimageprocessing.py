import numpy as np
#import PIL.Image
import tkinter as tk
import cv2
import PIL
import stackMain

from tkinter import *
from tkinter import ttk
from PIL import Image,ImageTk

def getMaskedImage(_maskMult):
    imageArray = np.array(openedImage) #convert image to numpy array #FORMAT SPECIFIC?
    #outpImageArray = rgb2gray( imageArray[...,0:3] ) # converts to greyscale
    
    #calculaste masking value for slider position and average brightness
    _maskValue = getArrayAverage(imageArray) * _maskMult/30
    
    maskArray = imageArray <= _maskValue
    imageArray[maskArray] = 0# sets values <= _mask_value to 0
    maskArray = imageArray > _maskValue
    imageArray[maskArray] = 255# sets values > _mask_value to 255
    
    outpImage = PIL.Image.fromarray(imageArray)# FORMAT SPECIFIC
    return outpImage

#Find average brightness of orignal image 
def getArrayAverage(_array): 
    return np.sum(_array) / _array.size

#Define a Function to update the image with currrent mask slider value
def updateImg(_maskMultiplier):
   maskMultiplier=float(_maskMultiplier)
   global maskedImage
   maskedImage = getMaskedImage(maskMultiplier) #call masking function

   #print("True")
   #Update GUI with new masked image
   global maskedImageTk
   maskedImageTk = ImageTk.PhotoImage( maskedImage )
   label.configure(image=maskedImageTk)
   #label.image=maskedImageTk
   #return masked_image

#outputs full res masked image
def stage2():
    global openedImage
    openedImage = openedImageRaw
    
    updateImg(slider.get())
    maskedImage.show()

    #close gui, without quit, kernel restarts
    win.quit()
    win.destroy()

#Calls focus stacking routine to stack images before working with   
stackMain.stackStart()

#Create an instance of tkinter frame
win= Tk()

#Load the image and resize for gui
openedImage = cv2.imread("merged.png", 0)
openedImageRaw = cv2.imread("merged.png", 0) #save a raw copy for final full res mask


imageDimensions = [int(openedImage.shape[1]/4),int(openedImage.shape[0]/4)] #find dimensions of image
openedImage = cv2.resize(openedImage, imageDimensions) #make image for gui usable size

#set gui window to fit image
winSize = (str(imageDimensions[0])+ "x" + str(imageDimensions[1]) + "1") 
win.geometry(winSize)


#Shows unmasked imaged before mask value changed
fromArray = Image.fromarray(openedImage)
unmaskedImage= ImageTk.PhotoImage(fromArray)
label= Label(win,image= unmaskedImage)
#label = Label(win, image = openedImage)
label.pack()


#Create a slider to vary image masking value
slider = tk.Scale(win, from_=0, to=100, command=updateImg, length=(500), orient=HORIZONTAL, label="Mask Threshold", tickinterval=20, resolution=0.5) 
slider.pack(pady=15)
win.bind("<Return>", updateImg) #update gui image when slider is moved

#Button to say when masking is complete
button = Button(win, text="Looks good", command=stage2)
button.pack()

win.call('wm', 'attributes', '.', '-topmost', '1')
win.mainloop()
    