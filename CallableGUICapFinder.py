import fromScratch

import sys
import numpy as np
import cv2
import tkinter as tk
import PIL
import copy
import scipy
from datetime import datetime as dt

from matplotlib import pyplot as plt
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from scipy.signal import find_peaks as fp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


###Finds capilaries from radius plot

###Section 1: smoothing radius data and finding gradient of its plot



###Section 2: GUI to adjust a gradient threshold for finding the capillaries,
#and buttons to manually delete miss-detections

def findCapEdges(capThreshold):
    global caps, showImage, resizedImage
    #resizedImg = resizedOriginal.copy()   
    
    #t=dt.now()  #timing peak finding
    
    capThreshold = int(capThreshold)
    #Finding capillaries from spiked in gradient of radial plot
    caps = []
    
    distanceSinceLast = 0
    
    for i in range (0,len(grads[0]-1)):
        if (distanceSinceLast>0.1):
            
            if (grads[1][i]>capThreshold):
                caps.append(grads[0][i])
                distanceSinceLast = 0
            if (grads[1][i]<-capThreshold):
                caps.append(grads[0][i])
                distanceSinceLast = 0
            
        distanceSinceLast = distanceSinceLast + 2*np.pi/capThreshold
        
    #print("Time to find intercepts: ", dt.now()-t)
    refreshButtonsAndPlot()

def refreshButtonsAndPlot():
    global centerX, centerY, caps, resizedImg, showImage, deleteButtons
    
    #Clears the previous delete button
    #Yes both lines are neccessary, one empties the list, one empties
    #the GUI grid
    removeAllButtons()
    deleteButtons.clear()
 
    for i in range (0, len(caps)):
         deleteButtons.append(Button(win, text=("Delete ",f"{caps[i]:.1f}"), command=lambda i=i: deleteEdge(i)))
         deleteButtons[i].grid(column=1, row=2+i)

    plotCapEdges(centerX, centerY, caps, resizedImg, showImage)
    
#Deletes the chosen detected edge
def deleteEdge(edgeIndex):
    global caps, deleteButtons
    caps.pop(edgeIndex)
   
    refreshButtonsAndPlot()
    
#Clear GUI grid of buttons for when capillaries are updated
def removeAllButtons():
    global deleteButtons
    for i in range (0,len(deleteButtons)):
        deleteButtons[i].destroy() 
        
#Calls the radius plotting function in fromScratch.py
def plotCapEdges(centerX, centerY, caps, resizedImg, showImage):   
    resizedImg = resizedOriginal.copy() 
    
    plot = fromScratch.plotPolarLines(centerX, centerY, caps, resizedImg, showImage)  
    #print("Time to recieve graph: ", dt.now()-t)
    
    resizedImg = cv2.resize(plot, imageDimensions)
    displayImage = PIL.Image.fromarray(resizedImg)
    displayImage = ImageTk.PhotoImage(displayImage)
    
    labelPlot.configure(image=displayImage)
    labelPlot.image=displayImage    
    
#Changes whether fibre image shown as background on capillary plot
#Setting to true slows down slider responsiveness
def showImageSwap():
    global showImage
    if showImage==False:
        showImage=True
    else:
        showImage=False
    
    #re-generated plot with/without image as background
    plotCapEdges(centerX, centerY, caps, resizedImg, showImage)
    
    
def finished():  
    win.destroy()
    global complete
    complete = True
    
    
def main(radPlot, inputImg, cX, cY):
    global complete, smoothedRads, grads, centerX, centerY, imageDimensions, win, label, labelPlot, resizedOriginal, resizedImg, showImage, capThreshold, deleteButtons
    
    complete = False
    print("Capillary edge finding started")
    while (complete==False):
        centerX, centerY = cX, cY #center coords need to be used globally in this program
        deleteButtons=[] #list of buttons for deleting corrosponding capillary edge detections
        
        #smooth input radius data, and find gradient of plot
        smoothedRads, grads = fromScratch.smoothRadsAndFindGrads(radPlot)
        maxGrads = max(grads[1])
        
        #Creating GUI window and setting the scale of the window and plot
        win = Tk()
        #set gui window to fit image
        screenWidth, screenHeight = win.winfo_screenwidth()*0.75, win.winfo_screenheight()*0.75
        win.geometry('%dx%d+0+0' % (screenWidth,screenHeight))
        
        scale = inputImg.shape[1]/(screenWidth*0.5)
        imageDimensions = [int(inputImg.shape[1]/scale),int(inputImg.shape[0]/scale)]
        #resizedImg = cv2.resize(inputImg, [int(inputImg.shape[1]/scale),int(inputImg.shape[0]/scale)])
        
        resizedOriginal = inputImg
    
        
        #Shows image before processing
        resizedImg = cv2.resize(inputImg, imageDimensions)
        fromArray = Image.fromarray(resizedImg)
        #fromArray.thumbnail(imageDimensions)
        tkImage= ImageTk.PhotoImage(fromArray)
        label = Label(win,image= tkImage)
        #label = Label(win, image = img)
        #label.pack()
        label.grid(column=0, row=1, rowspan=6)
        
        labelPlot = Label(win, image=tkImage)
        labelPlot.grid(column=0, row=1, rowspan=6)
        
        #Slider to select gradient threshold at which to declare the edge of a capillary
        capThreshold = Scale(win, from_=1, to=maxGrads,  length=(500), command=findCapEdges, orient=HORIZONTAL, label="Capillary edge threshold", resolution=1) 
        capThreshold.set(1000)
        capThreshold.grid(column=1, row=0)
    
        #Button to switch whether the fibre image is overlayed on the capillay plot
        #True is slower, but gives better visualisation
        showImage = False
        showImageButton = Button(win, text="Show fibre image (slows slider response)", command=showImageSwap)
        showImageButton.grid(column=1, row=1)
        
        #Button to say when the correct capillary edges are found
        circleDataButton = Button(win, text="Finished", command=finished)
        circleDataButton.grid(column=2, row=0)
        
        win.mainloop()
        
    print("Capillary edge finding finished")
    return caps
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    