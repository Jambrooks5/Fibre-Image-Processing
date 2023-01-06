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

#funcChat=dict()
#funcChat['updateImg']=[]

def updateImg(event):    
    #print("Updating image")
    resizedImg = resizedOriginal.copy()
    
    rows = resizedImg.shape[0]
    global circles
    circles = cv2.HoughCircles(resizedImg, cv2.HOUGH_GRADIENT, 2, int(seperationSlider.get()), param1=int(p1Slider.get()), param2=int(p2Slider.get()), minRadius=minRadSlider.get(), maxRadius=maxRadSlider.get())
    
    resizedImg = cv2.cvtColor(resizedImg,cv2.COLOR_GRAY2RGB)
    
    if circles is not None:
        circles = np.int16(np.around(circles))
        for i in circles[0, :]:
            center = (int(i[0]), int(i[1]))
            # circle center
            cv2.circle(resizedImg, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = (i[2])
            cv2.circle(resizedImg, center, radius, (255, 0, 0), 3)
    
    resizedImg = cv2.resize(resizedImg, imageDimensions)
    displayImage = PIL.Image.fromarray(resizedImg)
    #displayImage.thumbnail(imageDimensions)
    displayImage = ImageTk.PhotoImage(displayImage)
    label.configure(image=displayImage)
    label.image=displayImage#.grid(column=3, row=0)
    #label.image.pack(side=LEFT)
    #label.grid(column=0, row=3, columnspan=2)
    #plt.figure(dpi=200)
    #plt.imshow(img)
    #plt.show


def finished():  
    win.quit()
    win.destroy()
    global complete
    complete = True
    
    
def main(passedImage, preMasked):
    print("Circle finding started")
    
    global imageDimensions, scale, win, label, complete, inputImg, resizedImg, resizedOriginal, p1Slider, p2Slider, seperationSlider, minRadSlider, maxRadSlider
    
    complete = False
    
    while (complete==False):
        win = tk.Toplevel()
        
        #set gui window to fit image
        screenWidth, screenHeight = win.winfo_screenwidth()*0.75, win.winfo_screenheight()*0.75
        
        win.geometry('%dx%d+0+0' % (screenWidth,screenHeight))
        
        #winSize = (str(imageDimensions[0]*2)+ "x" + str(imageDimensions[1] + 600)) 
        #win.geometry(winSize)
        
        if (preMasked == False):
            inputImg = cv2.imread(passedImage)
            inputImg = cv2.cvtColor(inputImg,cv2.COLOR_BGR2GRAY)
            inputImg = cv2.medianBlur(inputImg, 5)
            
        else:
            #Swap comments to change input picture from np.array to image
            #inputImg = np.array(passedImage)
            #inputImg = cv2.medianBlur(inputImg, 5)
            inputImg = passedImage
            inputImg = cv2.medianBlur(inputImg, 5)
            
        #scale = 1/7
        
        #imageDimensions = [int(inputImg.shape[1]*scale),int(inputImg.shape[0]*scale)] #find dimensions of image
        #resizedImg = cv2.resize(inputImg, imageDimensions) #make image for gui usable size
        #resizedImg = inputImg
        
        scale = inputImg.shape[1]/(screenWidth*0.5)
        imageDimensions = [int(inputImg.shape[1]/scale),int(inputImg.shape[0]/scale)]
        #resizedImg = cv2.resize(inputImg, [int(inputImg.shape[1]/scale),int(inputImg.shape[0]/scale)])
        resizedImg = inputImg
        resizedOriginal = resizedImg
        
        
        #Shows image before processing
        resizedImg = cv2.resize(inputImg, imageDimensions)
        fromArray = Image.fromarray(resizedImg)
        #fromArray.thumbnail(imageDimensions)
        tkImage= ImageTk.PhotoImage(fromArray)
        label = Label(win,image= tkImage)
        #label = Label(win, image = img)
        #label.pack()
        label.grid(column=0, row=0, rowspan=6)
        
        
        p2Slider = Scale(win, from_=0, to=100,  length=(500), command=updateImg, orient=HORIZONTAL, label="Center Parameter", tickinterval=20, resolution=1) 
        #p2Slider.pack(pady=15)
        p2Slider.set(78)
        
        p1Slider = Scale(win, from_=0, to=500,  length=(500), command=updateImg, orient=HORIZONTAL, label="Canny edge detector", tickinterval=20, resolution=1) 
        #p1Slider.pack(pady=15)
        p1Slider.set(100)
        
        seperationSlider = Scale(win, from_=0, to=300,  length=(500), command=updateImg, orient=HORIZONTAL, label="Min seperation of circle centers", tickinterval=20, resolution=1) 
        #seperationSlider.pack(pady=15)
        seperationSlider.set(112)
        
        minRadSlider = Scale(win, from_=0, to=inputImg.shape[0],  length=(500), command=updateImg, orient=HORIZONTAL, label="Min radius of searched circles", tickinterval=20, resolution=1) 
        #minRadSlider.pack(pady=15)
        minRadSlider.set(20)
        
        maxRadSlider = Scale(win, from_=0, to=inputImg.shape[0],  length=(500), command=updateImg, orient=HORIZONTAL, label="Max radius of searched circles", tickinterval=250, resolution=1) 
        #maxRadSlider.pack(pady=15)
        maxRadSlider.set(170)
        
        p2Slider.grid(column=1, row=0)
        p1Slider.grid(column=1, row=1)
        seperationSlider.grid(column=1, row=2)
        minRadSlider.grid(column=1, row=3)
        maxRadSlider.grid(column=1, row=4)
        
        
        circleDataButton = Button(win, text="Finished", command=finished)
        #circleDataButton.pack(pady=15)
        circleDataButton.grid(column=1, row=5)
        
        #win.bind("<Any>", updateImg) #update gui image when slider is moved
        #win.call('wm', 'attributes', '.', '-topmost', '1')
        win.mainloop()
    
    print("Circle finding finihed")
    return circles


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

