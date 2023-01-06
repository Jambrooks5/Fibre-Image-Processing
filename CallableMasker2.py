import numpy as np
#import PIL.Image
import tkinter as tk
import cv2
import PIL
#import stackMain

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
    
    outpImage = imageArray
    #outpImage = PIL.Image.fromarray(imageArray)# FORMAT SPECIFIC
    return outpImage

#Find average brightness of orignal image 
def getArrayAverage(_array): 
    return np.sum(_array) / _array.size

#Define a Function to update the image with currrent mask slider value
def updateImg(_maskMultiplier):
   maskMultiplier=float(_maskMultiplier)
   global maskedImage
   maskedImage = getMaskedImage(maskMultiplier) #call masking function

   #Update GUI with new masked image
   global maskedImageTk
   maskedImageTk = ImageTk.PhotoImage(PIL.Image.fromarray(maskedImage))
   label.configure(image=maskedImageTk)

#outputs full resolution masked image
def finalFullRes():
    
    global openedImage
    openedImage = openedImageRaw  #reloads the original res image to be masked
    updateImg(slider.get())

    #close gui, without quit, kernel restarts
    win.quit()
    win.destroy()
    
    global complete
    complete = True

#Calls focus stacking routine to stack images before working with   
def main(passedImage, parentWindow):
    global complete
    complete = False
    print("Masking started")
    while (complete==False):
        #print("while")
        global openedImage, openedImageRaw, label, slider, button, win
        
        #Create an instance of tkinter frame
        win= tk.Toplevel(parentWindow)
        #win.geometry("750x500")
        
        screenWidth, screenHeight = win.winfo_screenwidth()*0.75, win.winfo_screenheight()*0.75
        win.geometry('%dx%d+0+0' % (screenWidth,screenHeight))
        
        #Load the image and resize for gui 
        #swap comment pair if image isn't already opened
        #openedImage = cv2.imread(passedImage, 0)
        #openedImageRaw = cv2.imread(passedImage, 0) #save a raw copy for final full res mask
        openedImage = passedImage
        openedImageRaw = passedImage
        
        scale = openedImage.shape[0]/(screenWidth*0.5)
        openedImage = cv2.resize(openedImage, [int(openedImage.shape[1]/scale),int(openedImage.shape[0]/scale)])
            
        #Shows unmasked imaged before mask value changed
        fromArray = Image.fromarray(openedImage)
        unmaskedImage= ImageTk.PhotoImage(fromArray)
        label= Label(win,image= unmaskedImage)
        label.grid(column=0, row=0, rowspan=2)
        
        #Create a slider to vary image masking value
        slider = tk.Scale(win, from_=0, to=256, command=updateImg, length=(500), orient=HORIZONTAL, label="Mask Threshold", tickinterval=20, resolution=0.5) 
        slider.grid(column=1, row=0)
        #win.bind("<Return>", updateImg) #update gui image when slider is moved
        
        #Button to say when masking is complete
        button = Button(win, text="Looks good", command=finalFullRes)
        button.grid(column=1, row=1)
        
        #win.mainloop()
      
    print("Masking finished")
    return maskedImage
