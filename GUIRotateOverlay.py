import cv2
import tkinter as tk
from tkinter import *
from tkinter import ttk
import fibreAnalyser
import generalFunctions as gf
import numpy as np
import PIL

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

dataD = {}
guiD = {}

####Enter picture names here####
dataD['img1'] = cv2.imread("KUV.jpg",0)
dataD['img2'] = cv2.imread("KUVtwist.jpg",0)

def updateImg(angle):
    guiD['axes'].cla()
    guiD['axes'].axis('off')
    
    guiD['axes'].imshow(dataD['img1'], cmap='Blues', alpha=0.7, extent=[-dataD['centers'][0][1], dataD['img1'].shape[1]-dataD['centers'][0][1], dataD['centers'][0][0]-dataD['img1'].shape[0], dataD['centers'][0][0]])

    rotationMatrix=cv2.getRotationMatrix2D(dataD['centers'][1][-1::-1], int(angle), 1.)
    dataD['rotatedChannel']=cv2.warpAffine(dataD['img2'], rotationMatrix, dataD['img2'].shape[-1::-1] )

    guiD['axes'].imshow(dataD['rotatedChannel'], cmap='Reds', alpha=0.4, extent=[-dataD['centers'][1][1], dataD['rotatedChannel'].shape[1]-dataD['centers'][1][1], dataD['centers'][1][0]-dataD['rotatedChannel'].shape[0], dataD['centers'][1][0]])

    guiD['figureCanvas'].draw()

    
def getCenter(img):
    print("Finding center of: ", img)
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

def cwRotate():
    angle = guiD['angleSlider'].get() + 1
    guiD['angleSlider'].set(angle)
    updateImg(angle)
    
def ccwRotate():
    angle = guiD['angleSlider'].get() - 1
    guiD['angleSlider'].set(angle)
    updateImg(angle)

def main():
    scale = 3

    #Read in images to be overlayed
    #dataD['img1'] = cv2.imread("KUV.jpg",0)
    #dataD['img2'] = cv2.imread("KUVtwist.jpg",0)
    
    #dataD['img1'] = cv2.imread("KUV.jpg",0)
    #dataD['img2'] = cv2.imread("KUVtwist.jpg",0)
    
    dataD['img1'] = cv2.resize(dataD['img1'], (int(dataD['img1'].shape[1]/scale), int(dataD['img1'].shape[0]/scale)))
    dataD['img2'] = cv2.resize(dataD['img2'], (int(dataD['img2'].shape[1]/scale), int(dataD['img2'].shape[0]/scale)))


    #Find the center of the core in both images
    #dataD['centers'] = [[1506.849315068493, 2082.1917808219177], [1235.9178082191781, 2084.8310502283107]]
    dataD['centers'] = [getCenter(dataD['img1']), getCenter(dataD['img2'])]
    print("centers: ", dataD['centers'])

    guiD['win'] = Tk() #initiate Tkinter window
    
    #Make a plot that is in a canvas in the GUI window
    guiD['figure'] = Figure(figsize=(20,15))
    guiD['figureCanvas'] = FigureCanvasTkAgg(guiD['figure'])
    NavigationToolbar2Tk(guiD['figureCanvas'])
    guiD['axes'] = guiD['figure'].add_subplot()
    
    #Put the first image on the plot, with 50% opacity
    #guiD['axes'].imshow(dataD['img1'], cmap='Blues', alpha=0.5, extent=[-dataD['centers'][0][1], dataD['img1'].shape[1]-dataD['centers'][0][1], dataD['centers'][0][0]-dataD['img1'].shape[0], dataD['centers'][0][0]])

    guiD['angleSlider'] = Scale(guiD['win'], from_=-180, to=180, command=updateImg, resolution=1, label="Rotation angle (degrees)", orient=HORIZONTAL, length=500)
    guiD['angleSlider'].set(0)
    guiD['angleSlider'].pack()
    
    guiD['ccwButton'] = Button(guiD['win'], text="<-", command=ccwRotate)
    guiD['ccwButton'].pack()
    
    guiD['cwButton'] = Button(guiD['win'], text="->", command=cwRotate)
    guiD['cwButton'].pack()
    
    guiD['figureCanvas'].get_tk_widget().pack()
    
    guiD['win'].mainloop()
    
main()























