#https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
import sys
import numpy as np
import cv2
import tkinter as tk
import PIL
import copy

from matplotlib import pyplot as plt
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk


def updateImg(event):    
    #print("Updating image")
    global centerX, centerY
    
    resizedImg = resizedOriginal.copy()
    
    rows = resizedImg.shape[0]
   
    resizedImg = cv2.cvtColor(resizedImg,cv2.COLOR_GRAY2RGB)
     
    #storing position of mouse when clicked
    centerX = int(scale*event.x)
    centerY = int(scale*event.y)
    
    pos = (centerX, centerY)
    #print(pos, centerX, centerY)
    # circle center
    cv2.circle(resizedImg, pos, 10, (0, 255, 0), 3)

    resizedImg = cv2.resize(resizedImg, imageDimensions)
    displayImage = PIL.Image.fromarray(resizedImg)

    displayImage = ImageTk.PhotoImage(displayImage)
    label.configure(image=displayImage)
    label.image=displayImage

    
def finished(event):  
    print("manualCenterFinding.finished")
    win.quit()
    win.destroy()
    global complete
    complete = True
    
    
def main(passedImage):
    print("Manual center finding started")
    
    global imageDimensions, scale, win, label, complete, inputImg, resizedImg, resizedOriginal
    
    complete = False
    
    while (complete==False):
        #Swap bellow comments depending if you're calling this program from another GUI
        #win = tk.Toplevel() #Use when opened alongside another GUI
        win = Tk() #Use when calling with no other GUIs open
        
        #set gui window to fit image
        screenWidth, screenHeight = win.winfo_screenwidth()*0.75, win.winfo_screenheight()*0.75
        
        win.geometry('%dx%d+0+0' % (screenWidth,screenHeight))
        

        inputImg = passedImage
                
        scale = inputImg.shape[1]/(screenWidth*0.5)
        imageDimensions = [int(inputImg.shape[1]/scale),int(inputImg.shape[0]/scale)]
        #resizedImg = cv2.resize(inputImg, [int(inputImg.shape[1]/scale),int(inputImg.shape[0]/scale)])
        resizedImg = inputImg
        resizedOriginal = resizedImg
        
        
        #Shows image before processing
        resizedImg = cv2.resize(inputImg, imageDimensions)
        fromArray = Image.fromarray(resizedImg)
        tkImage= ImageTk.PhotoImage(fromArray)
        label = Label(win,image= tkImage)
        label.grid(column=0, row=0, rowspan=6)
        
        text = Label(win, text="Press enter to confirm the rough center location.")
        text.config(font=("Courier",16))
        text.grid(column=1, row=0)
        
        win.bind("<Return>", finished) 
        
        win.bind("<Button 1>", updateImg) #update gui image when slider is moved
        #win.call('wm', 'attributes', '.', '-topmost', '1')
        win.mainloop()
    
    print("Manual center finding finished")
    return centerX, centerY


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

