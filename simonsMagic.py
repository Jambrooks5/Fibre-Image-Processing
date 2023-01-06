# Gaussian blur before scaling down?
# gradually decrease(\increase) binary threshold and moniter change in contour sizes?; may indicate when the edges stop being revealed and the noise starts; use the brightness distribution to set the bar;
# alternative to ↑ is calculating the correct threshold just using the brightness
# could allow some noise but only select the larger contours as in "motion-tracker.py"?;

import cv2, inspect
import numpy as np
#import pandas as pd
from datetime import datetime as dt
from matplotlib import pyplot as plt

####	PARAMTERS

#imageFileName="untapered.png"
imageFileName="cane1.jpg"
#imageFileName="cane4.jpg"

processingBlurDims=(3,3)

displaySizeLimsAuto=[1000,1000]# the diplayed channels will be resized to fit in a box with these dimensions, without changing the aspect ratio
numProcessedPixLimAuto=2**19

####	TEMPORARY PARAMETERS

thresholdMult=0.05 # the position of the binary threshold as a fraction of gradChannel.max() - gradChannel.min()
brightnessMode=3 # 0:centerPixel, 1:maxInSection, 2:averageOfSection, 3:min

####	IMAGE GRADIENT FUNCTIONS (EDGE IMAGES)

#↓ performs a modified (fractional) sobel operation, ie gradient/brightness; _channel must be greyscale, with float elements
def getModifiedSobel(_channel,_kernel):
	printLog("start")
	outp=np.zeros(_channel.shape, dtype=float)# ndarray.shape is the dimensions of the array, ie =(xLen,yLen)
	kernel=_kernel/np.sum(np.absolute(_kernel))# normalises the kernal to give it magnitude=1, so it never outputs a value outside the pixel brightness range, ie (0,255) or (0.0,1.0)
	for iY in range(0,_channel.shape[0]-kernel.shape[0]):
		for iX in range(0,_channel.shape[1]-kernel.shape[1]):# note this leaves the outer edge outp pixels black
			thisPixPos=(iY+int(kernel.shape[1]/2) ,iX+int(kernel.shape[0]/2))# pix=>pixel
			thisPix=0
			thisSectionBrightness=0
			#for iKX in range(0,kernel.shape[1]): for iKY in range(0,kernel.shape[0]): thisPix+=_channel[iY+iKY][iX+iKX]*kernel[iKY][iKX]; thisSectionBrightness += _channel[iY+iKY][iX+iKX];
			thisSection=_channel[ iY:iY+kernel.shape[1], iX:iX+kernel.shape[0] ]
			thisPix=abs(np.sum(np.multiply(thisSection,kernel) ) )
			#	↓ temporary, we must choose 1
			if brightnessMode==0: thisSectionBrightness=_channel[thisPixPos[0]][thisPixPos[1]]# noise: 1, lighting: 1
			elif brightnessMode==1: thisSectionBrightness=thisSection.max() # noise: .5, lighting: .5,always between 1 and 0, but reduces the %difference between edges and noise
			elif brightnessMode==2: thisSectionBrightness=np.sum(thisSection)/thisSection.size
			elif brightnessMode==3: thisSectionBrightness=thisSection.min() # noise: 2, lighting: 1; makes the max significantly higher than most walls, thresholdMult=~0.05 required
			#	↑ temporary, we must choose 1
			# ↓ converts to ~ the change in the fraction of the brightness
			thisPix/=thisSectionBrightness
			outp[thisPixPos[0]][thisPixPos[1]]=thisPix
	printLog("end")
	return outp

#↓ currently not used
def getModifiedSobel2(_channel):
	gradChannel=cv2.Laplacian(_channel,cv2.CV_64F)
	return np.divide(np.absolute(gradChannel),_channel)

#↓ ~ the fractional gradient
def getProcessedGradChannel(_channel, _blurKernalDims=(3,3), _numProcessedPixLim=0):# _numProcessedPixLim=0 means do not resize
	printLog("start")
	sobelX=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
	sobelY=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	bluredChannel=cv2.GaussianBlur(_channel,_blurKernalDims,0) ############################## blur removes noise but makes lighting changes more included 
	# scales down the channel (if necessary) to decrease the computation time
	if bluredChannel.size>_numProcessedPixLim and _numProcessedPixLim!=0:
		scaleMult=(_numProcessedPixLim/bluredChannel.size)**0.5
		newChannelDims=( int(bluredChannel.shape[1]*scaleMult), int(bluredChannel.shape[0]*scaleMult) )
		resizedChannel=cv2.resize(bluredChannel, newChannelDims)
	else: resizedChannel=bluredChannel
	#
	gradientXChannel=getModifiedSobel(resizedChannel,sobelX)
	gradientYChannel=getModifiedSobel(resizedChannel,sobelY)
	outp=np.sqrt(np.add( np.power(gradientXChannel,2), np.power(gradientYChannel,2) ))# (X^2+Y^2)^1/2; remove sqrt? helps to seperate walls from noise?
	outp=outp/outp.max() # normalises the channel
	#
	printLog("end")
	return outp

####	BINARY IMAGE CONVERSION FUNCTIONS

#↓ calculates the appropriate initial threshold value and threshold step size from the brightness distribution of the pixels
def getThresholdRangeVals(_channel):# WIP
	printLog("start")
	sortedPixels=np.sort(np.concatenate(_channel))# in ascending order of brightness; sorted may be unnecessary
	pixHist,pixBins=np.histogram(sortedPixels, bins=1000)# num_pixels_in_bins,edges_of_bins
	#	#
	printLog("end")
	return [initThreshold, stepSize]

def getWallToNoiseCheck(_contours,_heirarchy):# WIP
	contourAreas=np.array([ cv2.contourArea(iC) for iC in _contours ])
	#	#
	return outp

#↓ calculates the threshold value that seperates the fibre wall gradient from noise||lighting gradients; for getBinaryChannel
def getThreshold(_channel):# WIP
	printLog("start")
	#	#	#	#	#	#	#	#	#	#
	#[threshold,stepSize]=getThresholdRangeVals(_channel)
	#while threshold>0:
		#binaryChannel=cv2.threshold(_channel, threshold, 1.0, cv2.THRESH_BINARY)[1]
		#contours,heirarchy = cv2.findContours(binaryChannel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		#if getWallToNoiseCheck(contours,heirarchy): break# noise starts appearing \ fibre walls stop appearing
		#
		#threshold-=thresholdStepSize
		#if threshold<=0: set error; break;
	#	#	#	#	#	#	#	#	#	#
	#pixelsDF = pd.DataFrame(allPixels, columns = ['var_name'] )# Converting array to pandas DataFrame
	#pixelsDF.plot(kind = 'density')
	#	#	#	#	#	#	#	#	#	#
	printLog("end")
	return _channel.min()+(thresholdMult*( _channel.max() - _channel.min() ))# ← temporary, uses temporary parameter thesholdMult, change to automatic

def getBinaryChannel(_channel):
	printLog("start")
	threshold=getThreshold(_channel)
	outp=cv2.threshold(_channel, threshold, 1.0, cv2.THRESH_BINARY)[1]
	printLog("end")
	return outp

#↓ for testing purposes
def plotNumPixVsBrightness(_channel):
	printLog("start")
	allPixels=np.sort(np.concatenate(_channel))# in ascending order of brightness
	#
	plt.hist(allPixels, bins=1000)
	plt.xlabel("brightness")
	plt.ylabel("number of pixels")
	plt.show()
	#
	plt.plot( allPixels, range(0,len(allPixels)), label="number of dimmer pixels", color='black', marker='', linestyle='-', linewidth=1, markersize=1 )
	plt.xlabel("brightness")
	plt.legend()
	plt.show()
	#
	return 1

####	IMAGE ANALYSIS FUNCTIONS

#↓ performs Fourier transform, and centers the origin; can also take images, but doesn't produce a very coherant output
def getFourierChannel(_channel):
	printLog("start")
	fourier = np.fft.fft2(_channel)
	centeredFourier = np.fft.fftshift(fourier)# shifts the origin of the fourier spectrum to the center of the image
	outp = np.abs(centeredFourier)
	outp=outp/outp.max()#20*np.log(np.abs(fshift))
	printLog("end")
	return outp

####	DISPLAY FUNCTIONS

#↓ resized the channel to fit it within a box of [width,height]=_sizeLims
def getResizedImages(_image,_sizeLims):
	printLog("start")
	image_dims=( _image.shape[1], _image.shape[0] )# (width,height)
	size_mult = min([ _sizeLims[0]/image_dims[0], _sizeLims[1]/image_dims[1] ])
	outp_image_dims = tuple([int(i_id*size_mult) for i_id in image_dims])
	printLog("end")
	return cv2.resize(_image, outp_image_dims)

#↓ displays all input images and waits for input to close them
def displayImages(_names,_images,_displaySizeLims=[0,0]):# _displeySizeLims=[0,0] displays original image sizes, otherwise fits each image in a box of [width, height]=_displaySizeLims 
	printLog("start")	
	for i in range(0,len(_images)):
		thisImage = _images[i] if _displaySizeLims==[0,0] else getResizedImages(_images[i],_displaySizeLims)
		cv2.imshow( _names[i], thisImage )
	while 1:
		pressed_key=cv2.waitKey(1)&0xFF
		if pressed_key==ord('q'): break
	cv2.destroyAllWindows()
	printLog("end")
	return 1

####	MISCELLANEOUS FUNCTIONS

#↓ prints the input message with both the time and the function stack (the functions it is running within)
def printLog(_message="currently running"):
	function_nest=[i_stack.function for i_stack in inspect.stack()][1:-1]# [1:-1] is excluding printLog function && <module>
	print_outp = str(_message)
	for i_func in function_nest: print_outp = "("+i_func+"): "+print_outp
	print_outp = "\n" + str(dt.now()) + "----" + print_outp
	print(print_outp)
	return 1

def getGradientCoords(_coords,_datumSpan=1):# _coords must be [x_array, y_array], and must be in order
	printLog("start")
	outp=np.zeros( (2,len(_coords[0])-_datumSpan), dtype=float )# [xArr, dy_by_dx_arr]
	for i in range(0,len(outp[0])):
		delX=(_coords[0][i+_datumSpan]-_coords[0][i])
		delY=(_coords[1][i+_datumSpan]-_coords[1][i])
		outp[0][i]=_coords[0][i]+(delX/2)
		outp[1][i]=delY/delX
	printLog("end")
	return outp

def getAveragedArray(_in, _aveSpan):# _averaging_span ie number of points to average over
	return np.array([ np.sum(_in[i:i+_aveSpan])/_aveSpan for i in range(0,len(_in)-_aveSpan) ])

def getAveragedCoords(_coords, _aveSpan):
	printLog("start")
	outp=np.zeros( (len(_coords),len(_coords[0])-_aveSpan), dtype=float)
	for i0 in range(0,outp.shape[0]):
		outp[i0]=getAveragedArray(_coords[i0], _aveSpan)
	printLog("end")
	return outp

####	MAIN

def main(inputImage):
    printLog("start")
    #
    #channel=cv2.imread(imageFileName,0)/255# note that 2nd_arg=0 converts to greyscale (single channel)
    channel=inputImage
    gradChannel=getProcessedGradChannel(channel, processingBlurDims, numProcessedPixLimAuto)
    #binaryChannel=getBinaryChannel(gradChannel)
    #fourierBinary=getFourierChannel(binaryChannel)
    #
    #plotNumPixVsBrightness(gradChannel)
    #plotNumPixVsBrightness(fourierBinary)
    #displayImages( ["channel","grad channel","binary channel","Fourier binary"], [channel,gradChannel,binaryChannel,fourierBinary], displaySizeLimsAuto )
    #printLog("end")
    gradChannel=np.floor(gradChannel*255)
    return gradChannel

#main()
