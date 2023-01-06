import sys,os
sys.path.append("c:\\users\\user\\miniconda3\\lib\\site-packages")
import CallableMasker
import CallableMasker2
import callableCircleFinder
import simonsMagic
import fromScratch
import CallableGUICapFinder
import CallableGUICapFinder2
import manualCenterFinder
import plotFitting

import numpy as np
import cv2
import tkinter as tk
from tkinter import *
import PIL
import statistics as stat

from matplotlib import pyplot as plt

#Dictionary to hold all variables that are passed around functions
dataD = {}

#Dictionary to hold elements of GUI
guiD = {}

#IMPORTANT - conversion value from pixels to microns (pixels per micron)
# Only used when outputting values, not calculations
ppm = 29
#ppm = 1.45
###Select the picture of the fibre to analyse here###
dataD['originalImage'] = cv2.imread("KUV.jpg",0)


'''
###Captures an image on a conected microscope
def getCapturedImage():
    os.system("python captureImage.py")
    print("waiting for responce")
    while not os.path.isfile("_img_complete_indicator_.txt"): pass
    os.remove("_img_complete_indicator_.txt")
    return cv2.imread("_img_.png")
cv2.imshow("test",getCapturedImage())
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

###Masking###
def chooseMask():
    print("Started masking GUI")
    #dataD['originalImage'] = cv2.imread("Kflat.jpg",0) #open and store original image
    #dataD['originalImage']=cv2.GaussianBlur(dataD['originalImage'],(51,51),0)
    #Buttons which call either basic or sobel/gradient masking
    guiD['basicMaskButton'] = Button(guiD['mainWin'], text="Basic mask", command=basicMask)
    guiD['basicMaskButton'].pack()
    
    guiD['fancyMaskButton'] = Button(guiD['mainWin'], text="Fancy mask", command=fancyMask)
    guiD['fancyMaskButton'].pack()
    
6
def basicMask():
    print("basicMask called")
    #Remove masking buttons from GUI
    guiD['basicMaskButton'].destroy()
    guiD['fancyMaskButton'].destroy()
    
    #Call masking program
    dataD['maskedImage'] = CallableMasker.main(dataD['originalImage'])

    chooseCenterOrCircles() #Call next part of the GUI
 
def fancyMask():
    print("fancyMaskCalled")
    
    #Remove masking buttons from GUI
    guiD['basicMaskButton'].destroy()
    guiD['fancyMaskButton'].destroy()
    
    #Call sobel/gradient masking program, then pass that image to the basic masking program
    sobelImage = simonsMagic.main(dataD['originalImage'])
    sobelImage = np.array(sobelImage.astype(np.uint8))
    dataD['maskedImage'] = CallableMasker.main(sobelImage)

    chooseCenterOrCircles() #Call next part of the GUI
    
    '''
def finishMasking():
    #Remove masking options from GUI
    guiD['basicMaskButton'].destroy()
    guiD['fancyMaskButton'].destroy()
    #PIL.Image.fromarray(dataD['maskedImage']).show()
    print("Ended masking GUI")
    
    #Call circle/center selection
    chooseCenterOrCircles()
    '''
###Circle finding or manual center selection###
def chooseCenterOrCircles():
    print("Started center/circle GUI")
    guiD['textMessage'] = Label(guiD['mainWin'], text="Selecting 'Manual center' will let you manually select the rough center of the fibre,\n which will then be used for a radial plot, then the capillary edge finder GUI will come up.\n 'Circle finder' can find either the core or capillaries depending on the parameters you use. \n If a single circle is found, it will be assumed as the core wall and the radius will be found.\n If multiple are found, they will be assumed as all capillaries, and their radii and symmetry will be found.")
    guiD['textMessage'].pack()
    
    #Buttons which call either circle finding to find capillaries/core, or manually select rough center of core
    guiD['circlesButton'] = Button(guiD['mainWin'], text="Circle finding", command=circleFinding)
    guiD['circlesButton'].pack()
    
    guiD['manualCenterButton'] = Button(guiD['mainWin'], text="Manually center", command=manualCenter)
    guiD['manualCenterButton'].pack()

def circleFinding():
    #Remove previous button from GUI
    guiD['circlesButton'].destroy()
    guiD['manualCenterButton'].destroy()
    
    print("Chosen circleFinding")
    #Call circle finding GUI program
    dataD['circleData'] = callableCircleFinder.main(dataD['maskedImage'], True)
    
    #Count how many circles were found
    numOfCircles = len(dataD['circleData'][0])
                       
    #If only 1 circle found, assume it to be the core wall
    if (numOfCircles == 1):
        #add core radius and center coords to data dictionary
        dataD['coreRad'] = dataD['circleData'][0][0][2]
        dataD['centerX'] = dataD['circleData'][0][0][0]
        dataD['centerY'] = dataD['circleData'][0][0][1]
        
        #Update GUI text to explain next step
        guiD['textMessage']["text"] = "Found 1 circle - assumed to be core wall.\n Would you like to print the core radius, or perform a radial plot of the core? \nThe latter will bring up the capillary edge detecting GUI."
        #guiD['textMessage'].pack()
        
        #Buttons to print core radius, or call radius plotting program
        guiD['printCoreRadButton'] = Button(guiD['mainWin'], text="Print core radius", command=lambda: printData(['coreRad']))
        guiD['printCoreRadButton'].pack()
        
        guiD['radPlotButton'] = Button(guiD['mainWin'], text="Make core radius plot", command=coreWallToRadPlot)
        guiD['radPlotButton'].pack()
        
    elif (numOfCircles >=1):
        guiD['textMessage']["text"] = "Found multiple circles - assumed to be capillaries.\n Their average position will be calculated and taken as the core center.\n The polar angle of each capillary center and the gaps between them will be calculated.\n "
        dataD['centerX'], dataD['centerY'] = fromScratch.capCenterFinder(dataD['circleData'])
        dataD['capAngles'], dataD['capGaps'] = fromScratch.capSymmetry(dataD['circleData'], dataD['centerX'], dataD['centerY'])
        printData(['centerX', 'centerY', 'capAngles', 'capGaps'])
        
        #find theoretical capillary gaps from number of caps found
        perfectCapGap = 2*np.pi/len(dataD['capAngles'])
        print("Standard dev. of gaps between capillary centers from theoretical value: ", stat.stdev(dataD['capGaps'], perfectCapGap))

    else:
        print("No circles detected, closing GUI")
        closeGUI()

###Deletes option buttons from finding core wall circle before calling for radius plot
def coreWallToRadPlot():
    guiD['printCoreRadButton'].destroy()
    guiD['radPlotButton'].destroy()

    radPlotCaller()

def manualCenter():
    #Remove previous button from GUI
    guiD['circlesButton'].destroy()
    guiD['manualCenterButton'].destroy()
    
    print("Chosen manualCenter")
    
    guiD['textMessage']["text"] = """Once a center has been selected, a radial plot of the core will be generated.\n 
    This will then be used to find automatically find a more accurate center to the fibre, at which point another radius plot will be generated.\n
    These can take up to ~10s each, depending on the resolution of the fibre inner diameter.\n
    After this you will be given the option to remove any spikes in the plot due to noise or breaks in the capillary/jacket wall \n
    The plot will be used to find capillary edges, and the GUI for this will appear.\n
    If the radial plot still has a sinusoidal curve, you'll want to recalculate the center and capillary \n 
    edges using the buttons that appear after you've completed the intial cap. edge finding.
    """

    dataD['centerX'], dataD['centerY'] = manualCenterFinder.main(dataD['maskedImage'])
    ###
    dataD['radPlotNum'] = 0
    ###
    radPlotCaller()

###Generate radius plot with current data
def radPlotCaller():
    print("radPlotCaller called")
    #Create the radial plot of the fibre core
    dataD['radPlot'], dataD['coreEdgePoints']= fromScratch.coreRadiusPlotter(dataD['centerX'], dataD['centerY'], dataD['maskedImage'])
        
    #Smooth the radial plot and take its gradient, which is then used to find the capillary edges
    #dataD['smoothedRads'], dataD['grads'] = fromScratch.smoothRadsAndFindGrads(dataD['radPlot'])
    smoothRadGradCaller()    
    
    #Display the radius plot so the user can judge whether the center coordinate was accurate enough
    plt.plot(dataD['radPlot'][0], dataD['radPlot'][1])
    plt.title("Radius plot")
    plt.show()
    
    ###
    print(dataD['centerX'], dataD['centerY'])
    if (dataD['radPlotNum'] < 1): #if this is the first radius plot of this image, automatically find a better center and remake the radius plot 
        recalculateCenterWithCaps()
    else:
        #Call the options to try and remove spikes 
        spikeRemovalChoice()
        
        #Call the function that brings up the cap edge finding GUI and store the results in the data dictionary
        #capEdgeCaller()
        
#Stores a smoothed radius plot, and the gradient of that plot
def smoothRadGradCaller():
    dataD['smoothedRads'], dataD['grads'] = fromScratch.smoothRadsAndFindGrads(dataD['radPlot'])
    
    
###Gives user option to remove spikes from radius plot
def spikeRemovalChoice():
    guiD['textMessage']["text"] = """Before looking for capillaries, are there any sharp spikes in the radius plot displayed?\n 
    If so, click 'De-spike'. This will show you a plot with the original and attempted de-spiked radius plots.\n
    You will then be able to choose to the use the original or de-spiked.\n
        If the radius plot has no big spiked, click "Looks good" to continue to the capillary finding GUI.
    """
    
    guiD['despikeButton'] = Button(guiD['mainWin'], text="De-spike", command=callDespiker)
    guiD['despikeButton'].pack()
    
    guiD['goodRadPlotButton'] = Button(guiD['mainWin'], text="Looks good", command=capEdgeCallerPrelim)
    guiD['goodRadPlotButton'].pack()
 
###Despikes the radius plot, then gives the user to use the despiked or original plot
def callDespiker():
    #Remove previous buttons from spikeRemovalChoice
    guiD['despikeButton'].destroy()
    guiD['goodRadPlotButton'].destroy()
    
    dataD['despikedRadPlot'] = fromScratch.despiker(dataD['radPlot'])
    
    guiD['textMessage']["text"] = """The de-spiked radius plot should now be visible.\n
    If it worked well, then click "Use de-spiked plot" to use it for finding capillary edges.\n
    If it broken the plot in some way, click "Use original plot", and you'll have to manually\n
        remove false edge detections in the edge finding GUI.
    """
    
    #Plot the original and despiked radius plots
    plt.plot(dataD['radPlot'][0], dataD['radPlot'][1])
    plt.plot(dataD['despikedRadPlot'][0], dataD['despikedRadPlot'][1])
    plt.show()
    
    #Buttons to let user choose whether to use despiked or orignal radius plot
    guiD['useDespikedButton'] = Button(guiD['mainWin'], text="Use de-spiked plot", command=useDespiked)
    guiD['useDespikedButton'].pack()
    
    guiD['useOriginalButton'] = Button(guiD['mainWin'], text="Use original plot", command=useOriginal)
    guiD['useOriginalButton'].pack()
    
#Deletes previous GUI buttons, changes default radPlot to despiked version, and calls capillary edge finding GUI
def useDespiked():
    #Delete previous buttons
    guiD['useDespikedButton'].destroy()
    guiD['useOriginalButton'].destroy()
    
    #Set the default radius plot to be the de-spiked one
    dataD['radPlot'] = dataD['despikedRadPlot']
    
    #Update the gradient plot to use the new de-spiked radius plot
    smoothRadGradCaller()
    
    #Call edge finding GUI, with radPlot set to the de-spiked one
    capEdgeCaller()

#As above, but keeps the original radPlot as it was
def useOriginal():
    #Delete previous buttons
    guiD['useDespikedButton'].destroy()
    guiD['useOriginalButton'].destroy()

    #Call edge finding GUI, with radPlot as the orignal
    capEdgeCaller()
    
###Deleted buttons from spikeRemovalChoice before calling capEdgeCaller
def capEdgeCallerPrelim():
    #Remove previous buttons from spikeRemovalChoice
    guiD['despikeButton'].destroy()
    guiD['goodRadPlotButton'].destroy()
    
    capEdgeCaller()

###Call the capillary edge finding GUI with the current radius plot and center coordinates
def capEdgeCaller():
    print("capEdgeCaller called")
    
    #Call the cap finding GUI program with the previously found radius plot
    dataD['capEdges'] = CallableGUICapFinder2.main(dataD['smoothedRads'], dataD['grads'], dataD['maskedImage'], dataD['centerX'], dataD['centerY'])
    
    if (len(dataD['capEdges']) == 0):
        print("No capillary edges found, closing GUI")
        closeGUI()
    
    else:
        #guiD['textMessage']["text"] = "Lovely, now we've got some capillary edges.\n However, if the center was ."
        guiD['textMessage']["text"] = "Lovely, now we've got some capillary edges.\n However, if the center coordinate wasn't accurate, these edges won't be.\n Look at the previous radial plot, if there is a sinusoidal wave to the higher points, select 'Recalculate center'.\n This'll find a more accurate center, and find the capillary edges again.\n If the higher points lie along a flat line, click 'Center looks good'."
 
        #Buttons to either find a better center and re-run the radius plot and cap edge finding, or continue with the current cap edges
        guiD['centerRecalcButton'] = Button(guiD['mainWin'], text="Recalculate center", command=recalculateCenter)
        guiD['centerRecalcButton'].pack()
        
        guiD['goodCenterButton'] = Button(guiD['mainWin'], text="Center looks good", command=prelimCapEdges)
        guiD['goodCenterButton'].pack()


def recalculateCenterWithCaps():
    ###
    dataD['radPlotNum'] += 1
    
    print("recalculateCenterWithCaps")
    
    #Removes the capillary points from the radius plot without knowing capillary edges
    dataD['noCapRadPlot'] = fromScratch.roughCapExcluder(dataD['radPlot'])
    
    #Fits a sine curve to the core wall points in the radial plot, and store the amplitude, phase shift, and y-intercept
    dataD['sinParams'] = plotFitting.fitSinFunc(dataD['noCapRadPlot'])

    #Uses these sine curve parameters to calulate a new center - an perfect center would create a radius plot with no sinusoidal dependance
    dataD['centerX'], dataD['centerY'] = fromScratch.reCenter(dataD['centerX'], dataD['centerY'], dataD['sinParams'])

    #Re-calls the radius plot creater with the new center coordinates, which in turn bring up the capillary edge finder
    radPlotCaller()


###Uses the sine curve of the radial plot to find a more accurate center
def recalculateCenter():
    ###
    dataD['radPlotNum'] += 1
    
    print("recalculateCenter")
    
    #Remove previous button from GUI
    guiD['centerRecalcButton'].destroy()
    guiD['goodCenterButton'].destroy()
    
    guiD['textMessage']["text"] = "Recalculate center chosen. A new center will be calculate from the previous radius plot.\n The cap edge finding GUI will then show - these capillary edges should be more accurate"

    #Removes the capillary points from the radius plot
    dataD['noCapRadPlot'] = fromScratch.capExcluder(dataD['radPlot'], dataD['capEdges'])

    #Fits a sine curve to the core wall points in the radial plot, and store the amplitude, phase shift, and y-intercept
    dataD['sinParams'] = plotFitting.fitSinFunc(dataD['noCapRadPlot'])

    #Uses these sine curve parameters to calulate a new center - an perfect center would create a radius plot with no sinusoidal dependance
    dataD['centerX'], dataD['centerY'] = fromScratch.reCenter(dataD['centerX'], dataD['centerY'], dataD['sinParams'])

    #Re-calls the radius plot creater with the new center coordinates, which in turn bring up the capillary edge finder
    radPlotCaller()



###Rearrange the capEdge list to start with a leading edge, and check for false edge detections
def prelimCapEdges():
    print("prelimCapEdges chosen")
    
    #Remove previous buttons from GUI
    guiD['centerRecalcButton'].destroy()
    guiD['goodCenterButton'].destroy()
    
    #User message about checking capillary values for false detections
    #guiD['textMessage']["text"] = "Before using the edges for any calculations, the list will be checked for any false detections,\n characterised by consecutive leading or trailing edges. If you look at 'centerCaps' \n in the data dictionary, the second value for each edge should read 0,1,0,1,0,1... \n Not 0,1,1,0,1,0,1..."
    
    #If the first capillary edge detected is a 'trailing edge', this adds 2pi to it, and moves it to the end of the list
    #Makes finding the centers of capillaries and symmetry possible
    dataD['capEdges'] = fromScratch.capEdge0er(dataD['capEdges'])
    
    #Check if capillary edges have any false detections
    checkPassed = fromScratch.checkCapEdges(dataD['capEdges'])
    
    if (checkPassed == True):
        print("Edge checks passed")
        #guiD['textMessage']["text"] = "Nice, your capillary edges passed the very basic test. Now the center angle for each will be found, as well as the angle the angle between them."

        getFibreData()
        
    else:
        guiD['textMessage']["text"] = "The capillary edge values suggested a false detection or odd number of edges, so the GUI has been brought back.\n This was probably caused by a break in the fibre or capillary wall causing a jump in the radius plot.\n Use the 'delete *angle*' buttons to remove any detected edges that are not real"
        capEdgeCaller()
    

#Collates data about the capillaries from the edges
def getFibreData():
    print("getFibreData called")
    guiD['textMessage']["text"] = "Nice, your capillary edges passed the very basic test. Now for some data.\n Whilst you're reading this, the following will be calculated:\n Capillary center angles, \nangular spacing of capillaries, \n'core' diameter (inside the jacket),\n proper core diameter (smallest circle touching all capillaries), \nangular capillary widths.\n Press Next for options on further data."
    
    #Finds the angles of the center of each capillary, and the angular gaps between these
    dataD['capAngles'], dataD['capGaps'] = fromScratch.getCapAngles(dataD['capEdges'])
    
    #Find the center of the gap between each capillary
    dataD['gapCenters'] = fromScratch.gapCenters(dataD['capEdges'])

    #Recalculate the no-capillary radius plot
    dataD['noCapRadPlot'] = fromScratch.capExcluder(dataD['radPlot'], dataD['capEdges'])

    #Find the 'core' diameter (inside the jacket, but including capillaries)
    dataD['coreRad'] = fromScratch.coreRadFinder(dataD['noCapRadPlot'], dataD['capEdges'])
    
    #Find the 'guiding core' diameter (smallest circle that touches capillaries)
    dataD['guidingCoreRad'] = fromScratch.guidingCoreRadFinder(dataD['radPlot'], dataD['capAngles'])

    #Find the widths of each capillary from the edges
    dataD['capWidths'] = fromScratch.getCapWidths(dataD['capEdges'])
    
    #The radius of the average center of the capillaries, used when converting angular widths to linear widths
    capCenterRad = (dataD['coreRad'] + dataD['guidingCoreRad'])/2
  
    #Little data analysis / outputs
    print("HERE COMES THE DATA!")
 
    #Output number of found capillaries
    print("Number of capillaries: ", len(dataD['capAngles'])) 
    
    #Outputting the two core radii
    #print("Jacket core radius: ", int(dataD['coreRad']/ppm), " microns\nGuiding core radius: ", int(dataD['guidingCoreRad']/ppm), " microns") 
    print(f"Jacket inner diamter: {dataD['coreRad']*2/ppm:.1f} microns \nGuiding core diameter: {dataD['guidingCoreRad']*2/ppm:.1f} microns")
    
    #Taking difference in core radii as average capillary depth
    #print("Average capillary depth: ", int((dataD['coreRad']-dataD['guidingCoreRad'])/ppm), " microns")
    print(f"Average capillary depth: {(dataD['coreRad']-dataD['guidingCoreRad'])/ppm:.1f} microns")
    
    #Find standard deviation in angular widths of capillaries
    print(f"Average angular width of capillaries +- standard deviation: {stat.mean(dataD['capWidths']):.3f} \u00B1 {stat.stdev(dataD['capWidths']):.3f} radians")
    print(f"Average linear width of capillaries +- standard deviation: {stat.mean(dataD['capWidths'])*capCenterRad/ppm:.3f} \u00B1 {stat.stdev(dataD['capWidths'])*capCenterRad/ppm:.3f} microns")
    
    #find theoretical capillary gaps from number of caps found
    perfectCapGap = 2*np.pi/len(dataD['capAngles'])
    
    print(f"Standard dev. of gaps between capillary CENTERS from theoretical value: {stat.stdev(dataD['capGaps'], perfectCapGap):.3f}")

    #Find the linear distance between adjacent capillary walls (important for light leakage from core)
    dataD['claddingGaps'] = fromScratch.getCladdingGaps(dataD['coreRad'], dataD['guidingCoreRad'], dataD['capEdges'])
    
    print(f"Mean and std. distribution of the gap between adjacent capillaries: {stat.mean(dataD['claddingGaps'])/ppm:.2f} \u00B1 {stat.stdev(dataD['claddingGaps'])/ppm:.2f} microns")

    #Lets the user show a plot of these angles and 
    #TEXT NOT FINISHED
    #guiD['textMessage']["text"] = "The angles of the capillary centers have been found, along with the gaps between them.\n Would you like to plot these angles, or skip to...PUT SOMETHING HERE"

    guiD['furtherDataButton'] = Button(guiD['mainWin'], text="Next", command=furtherData)
    guiD['furtherDataButton'].pack()


#Lets user select what further data to calculate, depending on needs and quality of the image
def furtherData():
    guiD['furtherDataButton'].destroy()
    
    guiD['textMessage']["text"] = """'Print gaps between capillaries' will give you micron measurements for the gaps between each capillary.\n
    'Plot angles and gaps' will open an image showing the capillary angles, and the gaps between them.\n
    'Find capillary depths' will find the depth of the individual capillaries.\n
        This is only recommended if the jacket wall is continuous and reasonably smooth.
    """
    guiD['claddingGapsButton'] = Button(guiD['mainWin'], text="Print gaps between capillaries", command=showCladdingGaps)
    guiD['claddingGapsButton'].pack()
    
    guiD['showAnglesButton'] = Button(guiD['mainWin'], text="Plot angles and gaps", command=plotCapAngles)
    guiD['showAnglesButton'].pack()
    
    guiD['findCapDepthsButton'] = Button(guiD['mainWin'], text="Find capillary depths", command=findCapDepths)
    guiD['findCapDepthsButton'].pack()
    
    guiD['capRadPlotsButton'] = Button(guiD['mainWin'], text="Find capillary radius plots", command=getCapRadPlots)
    guiD['capRadPlotsButton'].pack()
    
#Prints the gaps between each capillary
def showCladdingGaps():
    print("Cladding gaps: ", [i/ppm for i in dataD['claddingGaps']])

    
#Calls the polar line plotter to show capillary center angles and gaps between them
def plotCapAngles():
    tempPlot = fromScratch.plotPolarLines(dataD['centerX'],dataD['centerY'],dataD['capAngles'],dataD['maskedImage'],True,dataD['capGaps'],False)
    PIL.Image.fromarray(tempPlot).show()

def findCapDepths():
    #Find the depth of each capillary (standoff from the jacket)
    dataD['capDepths'] = fromScratch.capDepths(dataD['capAngles'], dataD['gapCenters'], dataD['radPlot'])

    print(f"Capillaries have depths of: {dataD['capDepths']}")

#Gets a radius plot for the interior of each capillary
def getCapRadPlots():
    #Find the center coordinate of each capillary in the image
    #These coords are from the core radius plot, so their origin (0,0) is the center of the fibre, so the core center must be 
    #   added for the capillary rad plots
    dataD['capCenterCoords'] = fromScratch.getCapCenterCoords(dataD['capAngles'], dataD['coreRad'], dataD['guidingCoreRad'])

    #Get polar plot showing capillary angle and gaps, returned as a plot
    ax = fromScratch.plotPolarLinesPLOT(dataD['centerX'],dataD['centerY'],dataD['capAngles'],dataD['maskedImage'],dataD['capGaps'])

    #Adds circles to the center of each capillary
    for i in range(0,len(dataD['capCenterCoords'])):
        #print((int(dataD['capCenterCoords'][i][0]), int(dataD['capCenterCoords'][i][1])))
        
        #Use matplotlib default colours for circles so they corrospond to the radial plots
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        ax.add_patch(plt.Circle((int(dataD['capCenterCoords'][i][0]), int(dataD['capCenterCoords'][i][1])), dataD['maskedImage'].shape[1]/200, color=colors[i]))
        
    plt.show(ax)
    
    dataD['capRadPlots'] = []
    dataD['capBoundaries'] = []
    
    for i in range(0,len(dataD['capAngles'])):
        tempCapRadPlot, tempCapBoundaries = fromScratch.coreRadiusPlotter(dataD['capCenterCoords'][i][0]+dataD['centerX'], -dataD['capCenterCoords'][i][1]+dataD['centerY'], dataD['maskedImage'],100)
        #dataD['capRadPlots'].append(fromScratch.coreRadiusPlotter(dataD['capCenterCoords'][i][0]+dataD['centerX'], dataD['capCenterCoords'][i][1]+dataD['centerY'], dataD['maskedImage'],100)[0])    
        dataD['capRadPlots'].append(tempCapRadPlot)
        dataD['capBoundaries'].append(tempCapBoundaries)
        
        #Plot the capillary radius plots
        plt.figure(0)
        plt.title('Capillary interior radii plots')
        #plt.plot(tempCapRadPlot[0],tempCapRadPlot[1])
        
        despikedTempCapRadPlot = fromScratch.despiker(tempCapRadPlot)
        plt.plot(despikedTempCapRadPlot[0],despikedTempCapRadPlot[1])
        
        #Plots the boundaries of the capillaries, along with the capillary centers
        plt.figure(1)
        plt.axis('equal')
        plt.title('Detected capillary edges')
        plt.scatter(tempCapBoundaries[0],tempCapBoundaries[1], s=dataD['maskedImage'].shape[1]/300)
        plt.scatter(dataD['capCenterCoords'][i][0]+dataD['centerX'],-dataD['capCenterCoords'][i][1]+dataD['centerY'], color='k')
       
    #Show the two above plots   
    plt.figure(0)
    plt.show()
    
    plt.figure(1)
    plt.show()







###Print data fron data dictionary - pass in a list of keys as text, e.g. printData(['coreRad'])
def printData(keyList):
    for i in range(0,len(keyList)):
        print("\n", keyList[i], " = ", dataD[keyList[i]])



'''    
def finishCircles():
    print("Called finishCircles")
    guiD['chooseCirclesButton'].destroy()
    guiD['manualCenterButton'].destroy()
   
    print("Ended center/circle GUI")
'''    
    
def closeGUI():
    guiD['mainWin'].destroy()
    guiD['mainWin'].quit()

def main():
    guiD['mainWin'] = Tk()

    guiD['cancelButton'] = Button(guiD['mainWin'], text="Cancel", command=closeGUI)
    guiD['cancelButton'].pack()
    
    chooseMask()
    
    guiD['mainWin'].mainloop()
    


main()
#closeGUI()