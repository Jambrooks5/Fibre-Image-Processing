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

funcChat=dict()
funcChat['updateImg']=[]



def updateImg(event):    
    #print("Updating image")
    resizedImg = resizedOriginal.copy()
    
    rows = resizedImg.shape[0]
    global circles
    circles = cv2.HoughCircles(resizedImg, cv2.HOUGH_GRADIENT, 2, int(seperationSlider.get()), param1=int(p1Slider.get()), param2=int(p2Slider.get()), minRadius=minRadSlider.get(), maxRadius=maxRadSlider.get())
    
    resizedImg = cv2.cvtColor(resizedImg,cv2.COLOR_GRAY2RGB)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (int(i[0]), int(i[1]))
            # circle center
            cv2.circle(resizedImg, center, 1, (0, 100, 100), 1)
            # circle outline
            radius = (i[2])
            cv2.circle(resizedImg, center, radius, (255, 0, 0), 1)
    
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


def circleData():
    print(circles)
    circlesNum = circles.size/3
    
    centerXTotal = np.sum(circles[0,:,0])
    centerX = int(centerXTotal / circlesNum)
    centerYTotal = np.sum(circles[0,:,1])
    centerY = int(centerYTotal / circlesNum)
    
    print (centerX, centerY)
    
    
    cv2.circle(resizedOriginal, (centerX, centerY), 10, (255,0,0), 1)
    PIL.Image.fromarray(resizedOriginal).show()
    
win = Tk()

#set gui window to fit image
screenWidth, screenHeight = win.winfo_screenwidth(), win.winfo_screenheight()

win.geometry('%dx%d+0+0' % (screenWidth,screenHeight))

#winSize = (str(imageDimensions[0]*2)+ "x" + str(imageDimensions[1] + 600)) 
#win.geometry(winSize)


inputImg = cv2.imread("test2.png")
inputImg = cv2.cvtColor(inputImg,cv2.COLOR_BGR2GRAY)
inputImg = cv2.medianBlur(inputImg, 5)

#scale = 1/7

#imageDimensions = [int(inputImg.shape[1]*scale),int(inputImg.shape[0]*scale)] #find dimensions of image
#resizedImg = cv2.resize(inputImg, imageDimensions) #make image for gui usable size
#resizedImg = inputImg

scale = inputImg.shape[0]/(screenWidth*0.5)
resizedImg = cv2.resize(inputImg, [int(inputImg.shape[1]/scale),int(inputImg.shape[0]/scale)])
resizedOriginal = resizedImg


#Shows image before processing
fromArray = Image.fromarray(resizedImg)
#fromArray.thumbnail(imageDimensions)
tkImage= ImageTk.PhotoImage(fromArray)
label = Label(win,image= tkImage)
#label = Label(win, image = img)
#label.pack()
label.grid(column=0, row=0, rowspan=5)


p2Slider = Scale(win, from_=0, to=100,  length=(500), orient=HORIZONTAL, label="Center Parameter", tickinterval=20, resolution=1) 
#p2Slider.pack(pady=15)
p2Slider.set(16)

p1Slider = Scale(win, from_=0, to=500,  length=(500), orient=HORIZONTAL, label="Canny edge detector", tickinterval=20, resolution=1) 
#p1Slider.pack(pady=15)
p1Slider.set(100)

seperationSlider = Scale(win, from_=0, to=300,  length=(500), orient=HORIZONTAL, label="Min seperation of circle centers", tickinterval=20, resolution=1) 
#seperationSlider.pack(pady=15)
seperationSlider.set(112)

minRadSlider = Scale(win, from_=0, to=resizedImg.shape[0],  length=(500), orient=HORIZONTAL, label="Min radius of searched circles", tickinterval=20, resolution=1) 
#minRadSlider.pack(pady=15)
minRadSlider.set(20)

maxRadSlider = Scale(win, from_=0, to=resizedImg.shape[0],  length=(500), orient=HORIZONTAL, label="Max radius of searched circles", tickinterval=20, resolution=1) 
#maxRadSlider.pack(pady=15)
maxRadSlider.set(45)

p2Slider.grid(column=1, row=0)
p1Slider.grid(column=1, row=1)
seperationSlider.grid(column=1, row=2)
minRadSlider.grid(column=1, row=3)
maxRadSlider.grid(column=1, row=4)


circleDataButton = Button(win, text="Print circle data", command=circleData)
#circleDataButton.pack(pady=15)
circleDataButton.grid(column=1, row=5)

win.bind("<Return>", updateImg) #update gui image when slider is moved
#win.call('wm', 'attributes', '.', '-topmost', '1')
win.mainloop()