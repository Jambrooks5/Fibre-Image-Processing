###Finds capilaries from radius plot and allows user to delete any false edge detections in a GUI

#Difference to no.1 - this takes smoothedRads and grads as arguments
#no.1 took raw rads, and called the smoothing and grad finding 

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


###Section 2: GUI to adjust a gradient threshold for finding the capillaries,
#and buttons to manually delete miss-detections

#Looks at the gradient plot and marks capillary edges any time there is a spike
#Also stores whether the spike is +ve (1) suggesting the trailing edge of a capillary
# or -ve (0) for the leading edge when scanning through polar theta
def findCapEdges(capThreshold):
    global caps, showImage, resizedImage
    #resizedImg = resizedOriginal.copy()   
    
    #t=dt.now()  #timing peak finding
    
    capThreshold = int(capThreshold)
    #Finding capillaries from spiked in gradient of radial plot
    caps = []
    '''
    #stop repeated capillary detections at adjacent points on the gradient plot
    distanceSinceLast = 0
    
    for i in range (0,len(grads[0]-1)):
        if (distanceSinceLast>0.1):
            
            if (grads[1][i]>capThreshold):
                #caps.append(grads[0][i])
                #caps[0].append(grads[0][i])
                #caps[1].append(1)
                caps.append([grads[0][i], 1])
                distanceSinceLast = 0
            if (grads[1][i]<-capThreshold):
                #caps.append(grads[0][i])
                #caps[0].append(grads[0][i])
                #caps[1].append(0)
                caps.append([grads[0][i], 0])
                distanceSinceLast = 0
            
        distanceSinceLast = distanceSinceLast + 2*np.pi/capThreshold
     '''   
    #for i in range (0,len(grads[0]-1)):
    i=0
    #Look through every point in the gradient plot
    while (i<len(grads[0])-1):
        #If the value is above the selected threshold, take that as the start of a peak
        if (grads[1][i] > capThreshold):
            peakStart = i #starting index of a peak
            #Count number of points before plot falls back below threshold
            while (grads[1][i] > capThreshold and i<len(grads[0])-1):
                i+=1
                
            peakEnd = i #end index of a peak
            
            #Peak of peak index in gradient plot
            #peak = grads[1].index(max(grads[1][peakStart:peakEnd]))
            peak = peakStart + np.argmax(grads[1][peakStart:peakEnd])

            #Add angle of edge to caps
            caps.append([grads[0][peak], 1])
            
      #Repeat the same for a negative spike, but with np.argmin     
        elif (grads[1][i] < -capThreshold):
            peakStart = i #starting index of a peak
            #Count number of points before plot falls back below threshold
            while (grads[1][i] < -capThreshold and i<len(grads[0])-1):
                i+=1
                
            peakEnd = i #end index of a peak
            
            #Peak of peak index in gradient plot
            #peak = grads[1].index(min(grads[1][peakStart:peakEnd]))
            peak = peakStart + np.argmin(grads[1][peakStart:peakEnd])

            #Add angle of edge to caps
            caps.append([grads[0][peak], 0])
        
        #If the point is not above the threshold, move on
        else:
            i+=1
            
    
    
    
    #print("Time to find intercepts: ", dt.now()-t)
    refreshButtonsAndPlot()

def refreshButtonsAndPlot():
    global centerX, centerY, caps, resizedImg, showImage, deleteButtons
    
    #Clears the previous delete button
    #Yes both lines are neccessary, one empties the list, one empties
    #the GUI grid
    removeAllButtons()
    deleteButtons.clear()
 
    for i in range (0, len(caps)-1):
         deleteButtons.append(Button(win, text=("Delete ",f"{caps[i][0]:.1f}"), command=lambda i=i: deleteEdge(i)))
         deleteButtons[i].grid(column=1, row=2+i)

    plotCapEdges(centerX, centerY, caps, resizedImg, showImage)
    
#Deletes the chosen detected edge
def deleteEdge(edgeIndex):
    global caps, deleteButtons
    #caps[0].pop(edgeIndex)
    #caps[1].pop(edgeIndex)
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

    #Polar plotter only takes list of angles, so need list of angles without leading/trailing marker
    capsToPlot = [item[0] for item in caps]
    plot = fromScratch.plotPolarLines(centerX, centerY, capsToPlot, resizedImg, showImage)  
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
    win.quit()
    win.destroy()
    global complete
    complete = True
    
    
def main(smoothRadialPlot, gradientPlot, inputImg, cX, cY):
    global complete, smoothedRads, grads, centerX, centerY, imageDimensions, win, label, labelPlot, resizedOriginal, resizedImg, showImage, capThreshold, deleteButtons
    
    #Cannot use parameters as global variable - so need to redefine to use in this program
    smoothedRads = smoothRadialPlot
    grads = gradientPlot
    
    complete = False
    print("Capillary edge finding started")
    while (complete==False):
        centerX, centerY = cX, cY #center coords need to be used globally in this program
        deleteButtons=[] #list of buttons for deleting corrosponding capillary edge detections
        
        #smooth input radius data, and find gradient of plot
        #smoothedRads, grads = fromScratch.smoothRadsAndFindGrads(radPlot)
        maxGrads = max(grads[1])
        
        #Creating GUI window and setting the scale of the window and plot
        win = tk.Toplevel()
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    