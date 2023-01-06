import plotFitting

import sys
import numpy as np
import cv2
import tkinter as tk
import PIL
import copy
import scipy
import statistics as stat
import bisect
from datetime import datetime as dt
import statistics as stat


from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from scipy.signal import find_peaks as fp
import scienceplots

plt.style.use('science')

#Number of points to use in radial plot - needed in different functions 
#numThetaPoints=1000

#Find the gaps between adjacent capillaries
def getCladdingGaps(coreRad, guidingCoreRad, capEdges):
    gapWidths = []
    #Radius of av. capillary center from fibre center
    gapRad = (coreRad + guidingCoreRad)/2
    
    for i in range(1,len(capEdges)-1, 2):
        tempGapWidth = capEdges[i+1][0] - capEdges[i][0] #Gap in radians
        #Convert gap to pixel distance
        pixelGap = gapRad * tempGapWidth
        gapWidths.append(pixelGap)
    
    #Because the last gap wraps around the polar coords. it is calculated differently
    tempCapWidth = (capEdges[-1][0] + (capEdges[0][0] - 2*np.pi))/2   
    pixelGap = gapRad * tempGapWidth
    gapWidths.append(pixelGap)

    return gapWidths

#Finds the "twice the third flattening" measure of the ovality of a capillary 
#   from its radius plot

def getOvality(passedCapRadPlot):
    radPlot = copy.deepcopy(passedCapRadPlot) #make copy of radius plot so original is not changed
    
    #Smooth the plot by averaging its coordinates over 5 points
    radPlot = getAveragedCoords(radPlot, 5)
    plt.plot(radPlot[0], radPlot[1])
    
    minRad = min(radPlot[1])
    maxRad = max(radPlot[1])
    
    #Use scipy to find peak indices
    peaks, _ = fp(radPlot[1])
    print(peaks)
    
    
    for i in range(0,len(peaks)):
        #convert from indices to angles to plot
        peaks[i] = radPlot[0][peaks[i]]
        plt.plot([peaks[i], peaks[i]], [minRad, maxRad])
    
    plt.show()
    return 0

'''
def getOvality(passedCapRadPlot):
    radPlot = copy.deepcopy(passedCapRadPlot) #make copy of radius plot so original is not changed
    #plt.plot(radPlot[0], radPlot[1])
    
    #Smooth the plot by averaging its coordinates over 5 points
    radPlot = getAveragedCoords(radPlot, 5)
    plt.plot(radPlot[0], radPlot[1])
    
    minRad = min(radPlot[1])
    maxRad = max(radPlot[1])
    
    
    #Find index of maximum point on radius plot
    maxIndex = np.argmax(radPlot[1])

    #The number of list indices in pi/2 radins along the x-axis
    #e.g. radPlot with 100 points -> indexSep of 25
    indexSep = int(len(radPlot[0])/4)
    
    #Finds the 2 peaks and 2 troughs of the radius plot, with the assumption that the 
    #   these will all be seperated by pi/2 radians
    peaks = []
    for i in range(0,4):
        #Find the index of peaks
        peakIndex = (maxIndex + i*indexSep)%len(radPlot[0])
        peakAngle = radPlot[0][peakIndex]
        #Stores the height of the peak/trough
        peaks.append(radPlot[1][peakIndex])
        
        plt.plot([peakAngle, peakAngle], [minRad, maxRad])

        
    #print(peaks)
    
    #The list of peaks/troughs will always go [peak, trough, peak, trough],
    #   so we can find the combined peak/trough distance, giving the long/short
    #   axes of the ellipse respectively.
    longAxis = peaks[0]+peaks[2]
    shortAxis = peaks[1]+peaks[3]    
    
    #Calculating ovality as a percentage difference in the major and minor axes
    ovality = 100*(longAxis-shortAxis)/longAxis
    plt.show()
    
    return ovality
    
'''    

#Smoothes spikes out of a radius plot, from jacket breaks or noise
#Works by looking for jumps in the plot that are 10% of the plots standard deviation or higher,
#   then looking at a set number of points after a jump.
#If the plot value quickly returns to that before the jump, that jump is taken as unwanted, and 
#   the value at the jump is set to that before it.
#If the values after the jump remain high/low for long enough, it is taken as a genuine jump, 
#   e.g. a capillary, and the following points are kept
def despiker(passedRadPlot):
    radPlot = copy.deepcopy(passedRadPlot) #make copy of radius plot so original is not changed
    
    checkPoints = int(len(radPlot[0])/20)
    
    dev = stat.stdev(radPlot[1]) #standard deviation of points from mean
    #Tolerance above which a spike is suspected, set as 5% of the 'average' radius
    tol = 0.1 * dev
    #print(dev, tol)
    
    #deleted = []
    i=1
    while (i<len(radPlot[0])-checkPoints):
        delete = False #default to delete the fault
        
        #if the next point is more than 5% higher
        if(radPlot[1][i+1]>radPlot[1][i]+tol):  
            for j in range(1,checkPoints+1): #look at the next checkPoints points
                #if any of these points are smaller than the first point + 1/5 of the tolerance
                if(radPlot[1][i+j] < (radPlot[1][i]+tol/3)):
                   delete = True
                   #deleted.append(["high", j])
        
        elif(radPlot[1][i+1]<radPlot[1][i]-tol):  
            for j in range(1,checkPoints+1):
                #if any of these points are smaller than the first point + 1/5 of the tolerance
                if(radPlot[1][i+j] > (radPlot[1][i]-tol/5)):
                   delete = True
                   #deleted.append(["low", j])
        
        #else:
         #   deleted.append(["Good",-1])
            
        #if the next point is found to be anomalous, set it to the value of the point before
        if (delete==True):
            radPlot[1][i+1] = radPlot[1][i]
    
        #deleted.append(delete)
        i += 1
        
    return radPlot
    


#Find coordinates for the center of capillaries using their angles and the two core radii
def  getCapCenterCoords(capAngles, cRad, gCRad):
    avR = (cRad+gCRad)/2 #Radius of the center of the capillaries
    
    capCenterCoords = []
    for i in range(0,len(capAngles)):
        #find x and y coord of center of each capillary with trig
        x = avR*np.cos(capAngles[i])
        y = avR*np.sin(capAngles[i])
        
        capCenterCoords.append([x,y])
        
    return capCenterCoords


#Finds the index of the closest value in a list to a target - from:
    #https://stackoverflow.com/questions/56335315/in-a-python-list-which-is-sorted-find-the-closest-value-to-target-value-and-its
def findClosestIndex(a, x):
    i = bisect.bisect_left(a, x)
    if i >= len(a):
        i = len(a) - 1
    elif i and a[i] - x > x - a[i - 1]:
        i = i - 1
    return (i)

#Finds the depths of the capillaries (standoff from the jacket)
def capDepths(capAngles, gapCenters, radPlot):
    capDepths = []

    for i in range(0, len(capAngles)):
        #centerIndex = radPlot[0].index(capAngles[i]) #find the radPlot index of the center of the capillary
        centerIndex = findClosestIndex(radPlot[0], capAngles[i]%(2*np.pi))
        #cwIndex = radPlot[0].index(gapCenters[i-1]%(2*np.pi))
        cwIndex = findClosestIndex(radPlot[0], gapCenters[i-1]%(2*np.pi))
        #ccwIndex = radPlot[0].index(gapCenters[i]%(2*np.pi))
        ccwIndex = findClosestIndex(radPlot[0], gapCenters[i]%(2*np.pi))
        
        #Find the radius from the center of core to the face of the capillary - taking the mean of three points on the capillary
        capRad = stat.mean([radPlot[1][centerIndex-1],radPlot[1][centerIndex],radPlot[1][centerIndex+1]])
    
        #Find the radius of the jacket wall on either side of the capillary, taking the mean of 3 points on either side
        cwRad = stat.mean([radPlot[1][cwIndex],radPlot[1][cwIndex-1],radPlot[1][cwIndex+1]]) #jacket wall clockwise of the capillary
        ccwRad = stat.mean([radPlot[1][ccwIndex],radPlot[1][ccwIndex-1],radPlot[1][ccwIndex+1]]) #jacket wall counter-clockwise of the capillary

        #Average the clockwise and counter-clockwise radii
        jacketRad = stat.mean([cwRad, ccwRad])
        
        #Take the difference between jacket and capillary radii and add to the depth list
        capDepths.append(jacketRad-capRad)
        
    return capDepths

#Finds the center of the gaps between each capillary
def gapCenters(capEdges):
    print("gapCenters called")
    gapCenters = []
    
    for i in range (1,len(capEdges)-1, 2):
        tempCenter = (capEdges[i][0] + capEdges[i+1][0])/2
        gapCenters.append(tempCenter)
    
    #Because the last gap wraps around the polar coords. it is calculated differently
    tempCenter = (capEdges[0][0] + (capEdges[-1][0] - 2*np.pi))/2
    gapCenters.append(tempCenter)
    
    return gapCenters

#Find the width of each capillary from their edge angles
def getCapWidths(capEdges):
    capWidths = []
    
    for i in range(0,len(capEdges)-1,2):
        tempWidth = capEdges[i+1][0]-capEdges[i][0]
        capWidths.append(tempWidth)
        
    return capWidths


#Find the radius of the guiding core (smallerst circle touching capillaries) 
# from a radius plot and capillary edges
def guidingCoreRadFinder(radPlot, capAngles):
    print("guidingCoreRadFinder called")
    radSum = 0 #sum of capillary center radii
    for i in range(0,len(capAngles)):
        #No mean taken as we're averaging over all the capillaries anyway
        radSum = radSum + radPlot[1][int((capAngles[i]%(2*np.pi))*1000/(2*np.pi))]
   
    return radSum/len(capAngles)
    

#Find the radius of the core (inside jacket) from a radius plot and capillary edges
def coreRadFinder(noCapRadPlot, capEdges):
    print("coreRadFinder called")

    avRad = sum(noCapRadPlot[1]) / len(noCapRadPlot[1]) #Find the average radius to get core radius

    return avRad


#Takes the list of capillary edges and find the polar angle of each center, and the angle between the centers
def getCapAngles(capEdges):
    print("getCapAngles called")
    capAngles = []
    capGaps = []
    
    #Take each capillary center angle to be the average of its two edges
    for i in range(0,len(capEdges), 2):
        tempAngle = (capEdges[i][0] + capEdges[i+1][0]) / 2
        capAngles.append(tempAngle)
    
    #Find the angular gap between each capillary center
    for i in range(0,len(capAngles)-1):
        tempGap = capAngles[i+1] - capAngles[i]
        capGaps.append(tempGap)
    
    #Because the last gap wraps around the polar coords. it is calculated differently
    tempGap = capAngles[0] - capAngles[-1] + 2*np.pi
    capGaps.append(tempGap)
    
    return capAngles, capGaps


#Checks the rearranged capillary edge values to check for any consecutive leading or trailing edges
# suggesting a break in the wall/capillary was falsely labeled as a capillary edge
#Also checks for an even number of capillary edges
def checkCapEdges(capEdges):
    #check for even number of capillary edges
    if (len(capEdges)%2 != 0):
        print("Odd number of capillary edges detected")
        return False
    
    #store previous edge direction (leading,0 or trailing,1) - starting values is trailing
    prev=1
    for i in range(0,len(capEdges)):
        if (capEdges[i][1] == prev):
            print("Error in capillary edges found, two consecutive edges were found to both be either leading or trailing")
            return False
        prev = capEdges[i][1]
        
    return True

###If the first capillary edge detected is a trailing edge, this function will 
# add 2pi to it and shift the list so the first edge is a leading edge.
#This format is thn easier to do calculations on for cap centers/widths etc.
def capEdge0er(capEdges):
    if (capEdges[0][1]==1): #If the first edge is a trailing edge
        #Add 2pi to that angle, and add it to the end of the list of edges
        capEdges.append([capEdges[0][0]+2*np.pi, capEdges[0][1]])
        #Remove the original angle from the start of the list
        capEdges.pop(0)
    
    return capEdges


###Uses the sin curved fitted to the core radius plot to find a more accurate center
# of the fibre
def reCenter(originalX, originalY, sinParams):
    #core radii at 0,pi/2,pi,3pi/2 from fitted sin curve
    rThetas = []
    
    for i in range(0,4):
        rThetas.append(plotFitting.sinFunc((i*2*np.pi/4), sinParams[0], sinParams[1], sinParams[2]))

    centerX = originalX + (rThetas[0]-rThetas[2])/2
    centerY = originalY - (rThetas[1]-rThetas[3])/2

    return centerX, centerY


###Find the closest capillary edge on either side of a given polar angle
#Stores the clockwise neighbour then counterclockwise as their index in the capEdges list
def adjacentCaps(theta, capEdges):
    foundCap = False
    capNum = 0
    
    while (foundCap==False and capNum<len(capEdges)):
        if (capEdges[capNum][0] > theta):
            foundCap = True
            return [capNum-1, capNum]
        elif (capNum==len(capEdges)-1):
            return [-1, 0]
            
        capNum += 1 


###Removes capillary areas from radius plot using the found capillary edges
def capExcluder(globalRadPlot, capEdges):
    #This deepcopy prevents the full radPlot being changed in the dictionary
    radPlot = copy.deepcopy(globalRadPlot)
    
    i=0
    
    #Removes data points bound by 2 capillary edges (those on the surface of the capillary)
    #Some points on the edges can remain after this
    while (i<len(radPlot[0])):
        #print(i)
        adjacentCapIndexes = adjacentCaps(radPlot[0][i], capEdges)

        if ((capEdges[adjacentCapIndexes[0]][1]==0 and capEdges[adjacentCapIndexes[1]][1]==1) or adjacentCapIndexes[0]==-1):
            radPlot[0].pop(i)
            radPlot[1].pop(i)
        else:
            i+=1

    #A few points near edges can survive the above - the below removes them
    
    #Fit sin curve to remaing 'core wall' points
    sinParams = plotFitting.fitSinFunc(radPlot)
    
    i=0
    #Go through all remaining points and exlude those far from the fitted line
    while (i<len(radPlot[0])-1):
        #Find error between radPlot value and fitted curve
        dataY = radPlot[1][i]
        curveY = plotFitting.sinFunc(radPlot[0][i], sinParams[0], sinParams[1], sinParams[2])
        fracError = abs((dataY-curveY)/curveY)

        #FIND PROPER NUMBER FOR THIS
        if (fracError>0.1):
            radPlot[0].pop(i)
            radPlot[1].pop(i)
        else:
            i+=1
              
    return radPlot

def roughCapExcluder(globalRadPlot):
    radPlot = copy.deepcopy(globalRadPlot)

    #Fit sin curve to remaing 'core wall' points
    sinParams = plotFitting.fitSinFunc(radPlot)
    
    i=0
    #Go through all remaining points and exlude those far from the fitted line
    while (i<len(radPlot[0])):
        #Find the radPlot value and fitted curve value 
        dataY = radPlot[1][i]
        curveY = plotFitting.sinFunc(radPlot[0][i], sinParams[0], sinParams[1], sinParams[2])

        #FIND PROPER NUMBER FOR THIS
        if (dataY<curveY):
            radPlot[0].pop(i)
            radPlot[1].pop(i)
        else:
            i+=1
              
    return radPlot
    
###Plots polar line as below, but returns result as a plot so it can be added to later
def plotPolarLinesPLOT(centerX, centerY, angles, plotImg, gaps=0):
    #Store the two end coordinates of each line at a time
    #One end is always the center 0,0
    xValues = [0,0]
    yValues = [0,0]
    #Length of drawn line indicating capillaries - exact value not important
    radius = np.array(plotImg).shape[1]/2

    f, ax = plt.subplots(dpi=400)
    
    #Runs through the angle each capillary is detected at,
    # and plots line to show it, along with the angular coord.
    for i in range(0,len(angles)):
        xValues[1] = int(radius*np.cos(angles[i]))
        yValues[1] = int(radius*np.sin(angles[i]))
    
        ax.plot(xValues, yValues, 'bo', linestyle="--")
        ax.text(xValues[1]/2, yValues[1]/2, f"{angles[i]:.1f}")
    
    #If the function was passed the gaps between angles, this still plot them 
    if (gaps!=0):
        for i in range (0, len(gaps)):
            xValues[1] = int(radius*np.cos(angles[i] + gaps[i]/2))
            yValues[1] = int(radius*np.sin(angles[i] + gaps[i]/2))
        
            ax.text(xValues[1]/4-radius/10, yValues[1]/4, f"{gaps[i]:.2f}")


    ax.axis('equal') #ensures plotting on sqaure grid

    ax.imshow(plotImg, cmap='Greys', alpha=0.5,  extent=[-centerX, plotImg.shape[1]-centerX, centerY-plotImg.shape[0], centerY], aspect='equal')

    return ax



###Plots polar line for any given angle, and for gaps between capillaries if passed
def plotPolarLines(centerX, centerY, angles, plotImg, showImage, gaps=0, returnAsImage=True):
    #print("plotPolarLines started")
    
    #Store the two end coordinates of each line at a time
    #One end is always the center 0,0
    xValues = [0,0]
    yValues = [0,0]
    #Length of drawn line indicating capillaries - exact value not important
    radius = np.array(plotImg).shape[1]/2

    #Canvas stuff allows plot to be converted to image for passing back
    #to CallableGUICapFinder
    fig = Figure(dpi=300)#figsize=(plotImg.shape[1]/100, plotImg.shape[0]/100))
    canvas = FigureCanvas(fig)
    polarPlot = fig.gca()
    
    polarPlot.axis('off')
    
    #Runs through the angle each capillary is detected at,
    # and plots line to show it, along with the angular coord.
    for i in range(0,len(angles)):
        xValues[1] = int(radius*np.cos(angles[i]))
        yValues[1] = int(radius*np.sin(angles[i]))
    
        polarPlot.plot(xValues, yValues, 'bo', linestyle="--")
        polarPlot.text(xValues[1]/2, yValues[1]/2, f"{angles[i]:.1f}")
    
    #If the function was passed the gaps between angles, this still plot them 
    if (gaps!=0):
        for i in range (0, len(gaps)):
            xValues[1] = int(radius*np.cos(angles[i] + gaps[i]/2))
            yValues[1] = int(radius*np.sin(angles[i] + gaps[i]/2))
        
            polarPlot.text(xValues[1]/4-radius/10, yValues[1]/4, f"{gaps[i]:.2f}")


    polarPlot.axis('equal') #ensures plotting on sqaure grid

    #Either does/doesn't show background image
    #Showing makes slider less responsive
    if showImage==True:
        #polarPlot.imshow(plotImg, cmap='Greys', alpha=0.5,  extent=[-centerX, plotImg.shape[1]-centerX, -centerY, plotImg.shape[0]-centerY], aspect='equal')
        polarPlot.imshow(plotImg, cmap='Greys_r', alpha=0.5,  extent=[-centerX, plotImg.shape[1]-centerX, centerY-plotImg.shape[0], centerY], aspect='equal')

    #origin='upper',
    #else:
     #   polarPlot.axis('equal') #ensures plotting on sqaure grid
        
    #Return either an image to be put in a GUI, or the raw plot whenever it's wanted
    #if returnAsImage==True:
        #Converts plot to image to then be passed
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(canvas.get_width_height()[::-1] + (3,))
    
    return image
    #else:
        
     #   return polarPlot
    


###find center position of detected capillaries
def capCenterFinder(circleData):
    print("capCenteFinder started")

    circleNum = circleData.size/3 #number of capillaries detected
    #Finds coordiante of center
    centerXTotal = np.sum(circleData[0,:,0])
    centerX = int(centerXTotal / circleNum)
    centerYTotal = np.sum(circleData[0,:,1])
    centerY = int(centerYTotal / circleNum)

    return centerX, centerY


###Find the polar angle of each capillary center, and the angles between the centers
def capSymmetry(circleData, centerX, centerY):
    print("capSymmetry started")

    angles = [] #polar coord angle of each capillary
    gaps = [] #radian angle between capillaries
    
    #subtract fibre center from capillary coords
    circleData[0,:,0], circleData[0,:,1] = circleData[0,:,0]-centerX, circleData[0,:,1]-centerY
    #print(circleData)
    #calculate polar angle of each capillary
    for i in range (0, int(circleData.size/3)):
        angle = np.arctan2(circleData[0,i,1], circleData[0,i,0])
        angles.append(angle)
        #print (angle)
    angles.sort() #sort angle into numerical order
    
    #find angluar gap between adjacent capillaries
    for i in range (0, int(circleData.size/3)-1):
        gaps.append(angles[i+1]-angles[i])
    
    #add gap that crosses polar axis
    gaps.append((np.pi-angles[len(angles)-1]) + (np.pi+angles[0]))
    
    return angles, gaps


###Plots radius out from the center before hitting glass wall
def coreRadiusPlotter(centerX, centerY, maskedImage, numThetaPoints=1000):
    print("coreRadiusPlotter started")
    
    #cv2.circle(maskedImage, (int(centerX),int(centerY)), 10, (0,255,0),3)
    #cv2.imshow(str(centerX), maskedImage)
    
    #numThetaPoints=1000
    radius = 1
    boundaryCoords=[[],[]]
    plotValues = [[],[]]
    centerColor = 0
    for i in range(0,numThetaPoints):
        theta=2*np.pi*i/numThetaPoints
        radius=1
        equalsCenter=True
        while (equalsCenter == True):
            pixel = [int(radius*np.cos(theta) + centerX), int(-radius*np.sin(theta) + centerY)]
            try:
                if maskedImage[pixel[1]][pixel[0]] != centerColor:
                    equalsCenter = False
            except IndexError:
                equalsCenter = False
            radius = radius + 1
        plotValues[0].append(theta)
        plotValues[1].append(radius)
        
        boundaryCoords[0].append(pixel[0])
        boundaryCoords[1].append(pixel[1])

    print("coreRadiusPlotter finished")
    return plotValues, boundaryCoords


'''Smoothing radial core plot and finding its gradient'''

###Smooths array of data over '_aveSpan' number of points
def getAveragedArray(_arr, _aveSpan):# _averaging_span ie number of points to average over
    return np.array([ np.average(_arr[i:i+_aveSpan]) for i in range(0,len(_arr)-_aveSpan) ])

###Smooths coordinates in multiple dimensions 
def getAveragedCoords(_coords, _aveSpan):
    outp=np.zeros( (len(_coords),len(_coords[0])-_aveSpan), dtype=float)
    for i0 in range(0,outp.shape[0]):
        outp[i0]=getAveragedArray(_coords[i0], _aveSpan)
    return outp
   
###Finds the gradient of a set of data/plot 
def getGradientCoords(_coords,_datumSpan=1):# _coords must be [x_array, y_array], and must be in order
    outp=np.zeros( (2,len(_coords[0])-_datumSpan), dtype=float )# [xArr, dy_by_dx_arr]
    for i in range(0,len(outp[0])):
        delX=(_coords[0][i+_datumSpan]-_coords[0][i])
        delY=(_coords[1][i+_datumSpan]-_coords[1][i])
        outp[0][i]=_coords[0][i]+(delX/2)
        outp[1][i]=delY/delX
    return outp

###Smoothes the radial core plot and finds its gradient at all point
def smoothRadsAndFindGrads(radPlot):
    smoothedRads = getAveragedCoords(radPlot, int(len(radPlot[0])/150))
    #plt.plot(smoothedRads[0], smoothedRads[1])
    #plt.show()
    
    grads=getGradientCoords(smoothedRads)
    #plt.plot(grads[0], grads[1])
    #plt.show()
    
    return smoothedRads, grads


