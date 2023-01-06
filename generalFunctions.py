import inspect, cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from datetime import datetime as dt


#		MISC


# TODO
def runCodeAsNewProgram(_codeString):
	return 1


#		ARRAY & COORDINATE FUNCTIONS


#
def getSecondMax(_array):
	maxIndex=0; secondMaxIndex=0
	for i in range(0,len(_array)):
		if _array[i]>_array[maxIndex]: secondMaxIndex=maxIndex; maxIndex=i
		elif _array[i]>_array[secondMaxIndex]: secondMaxIndex=i
	return secondMaxIndex

#
def getGradientCoords(_coords,_datumSpan=1):# _coords must be [x_array, y_array]
	printLog("start")
	sortIdicies=np.argsort(_coords[0])
	coords=[ [_coords[0][i] for i in sortIdicies], [_coords[1][i] for i in sortIdicies] ]
	outp=np.zeros( (2,len(coords[0])-_datumSpan), dtype=float )# [xArr, dy_by_dx_arr]
	for i in range(0,len(outp[0])):
		delX=(coords[0][i+_datumSpan]-coords[0][i])
		delY=(coords[1][i+_datumSpan]-coords[1][i])
		outp[0][i]=coords[0][i]+(delX/2)
		outp[1][i]=delY/delX if delX!=0 else np.nan
	printLog("end")
	return outp

#
def getAveragedArray(_arr, _aveSpan):# _averaging_span ie number of points to average over
	return np.array([ np.average(_arr[i:i+_aveSpan]) for i in range(0,len(_arr)-_aveSpan) ])

#
def getAveragedCoords(_coords, _aveSpan):
	printLog("start")
	outp=np.zeros( (len(_coords),len(_coords[0])-_aveSpan), dtype=float)
	for i0 in range(0,outp.shape[0]):
		outp[i0]=getAveragedArray(_coords[i0], _aveSpan)
	printLog("end")
	return outp

#
def getExtremaIndicies(_coords, _checkRange=1, _absoluteRangeMode=False):# _checkRange is 
	printLog("start")
	outp=[[],[]]# [minima,maxima]
	if _absoluteRangeMode==False:
		for i in range(_checkRange,len(_coords[0])-_checkRange):
			comparisons={ _coords[1][i]<_coords[1][i+i2] for i2 in range(-_checkRange,_checkRange+1) if i2!=0 }
			if comparisons=={True}: outp[0].append(i)# minima
			elif comparisons=={False}: outp[1].append(i)# maxima or plateau
	else:
		outp="WIP"
	printLog("end")
	return outp# [minima,maxima]

def getArrayCenterIndex(_arr):
	return tuple( int(.5*i-.5) for i in _arr.shape )

#
def getArrayMaxIndex(_arr):
	return np.unravel_index(np.argmax(_arr),_arr.shape)


####	PRINT FUNCTIONS


# prints the input message with both the time and the function stack (the functions it is running within)
def printLog(_message="currently running",*args,**kwargs):
	function_nest=[i_stack.function for i_stack in inspect.stack()][1:-1]# [1:-1] is excluding printLog function && <module>
	print_outp = str(_message)
	for i_func in function_nest: print_outp = "("+i_func+"): "+print_outp
	print_outp = "\n" + str(dt.now()) + "----" + print_outp
	print(print_outp,*args,**kwargs)
	return 1

#
def printUpdatePercentage(_currentVal,_maxVal=1,_decimalPlaces=1):
	message=str(round(100*_currentVal/_maxVal,_decimalPlaces)) + "%" + " "*9
	print(message,end="\r")
	return 1


#		IMAGE PROCESSING FUNCTIONS


# scales down the channel (if necessary) to decrease the computation time
def getSizeLimited(_channel,_numProcessedPixLim=0):
	if _channel.size>_numProcessedPixLim and _numProcessedPixLim!=0:
		scaleMult=(_numProcessedPixLim/_channel.size)**0.5
		newChannelDims=( int(_channel.shape[1]*scaleMult), int(_channel.shape[0]*scaleMult) )
		outp=cv2.resize(_channel, newChannelDims)
	else: outp=_channel
	return outp

def convert1To255(_channel): return np.multiply(_channel, 255).astype(np.uint8)

def convert255To1(_channel): return np.multiply(_channel.astype(float), 1/255)

def getNormalised(_channel):
	outp=np.add(_channel,-_channel.min())
	outp/=outp.max()
	return outp

def getInvChannel(_channel):
	return np.ones_like(_channel)-_channel.copy()

def getWeightedAveragePixelIndex(_channel):
	outp=[0,0]
	for iY in range(0,_channel.shape[0]):
		for iX in range(0,_channel.shape[1]):
			outp[0]+= _channel[iY][iX]*iY
			outp[1]+= _channel[iY][iX]*iX
	outp[0]=int(outp[0]/np.sum(_channel))
	outp[1]=int(outp[1]/np.sum(_channel))
	return outp


#		DISPLAY FUNCTIONS


#
def showImages(_names,_images,_displaySizeLims=[0,0]):# displays original image sizes if _displeySizeLims==[0,0] else fits each image in a box of [width, height]=_displaySizeLims 
	for i in range(0,len(_images)):
		thisImage = _images[i] if _displaySizeLims==[0,0] else getResizedImage(_images[i],_displaySizeLims)
		cv2.imshow( _names[i], thisImage )
	return 1

#
def cv2WaitToDestroy():
	printLog("start")	
	while 1:
		pressed_key=cv2.waitKey(0)&0xFF
		if pressed_key==ord('q'): break
	cv2.destroyAllWindows()
	printLog("end")
	return 1

# displays all input images and waits for input to close them
def displayImages(_names,_images,_displaySizeLims=[0,0]):# displays original image sizes if _displeySizeLims==[0,0] else fits each image in a box of [width, height]=_displaySizeLims 
	showImages(_names,_images,_displaySizeLims)
	cv2WaitToDestroy()
	return 1

#
def plotDataSequences(_names,_coordsList):
	colours=[]
	fig,ax=plt.subplots()
	for i in range(0,len(_names)):
		colour_=hsv_to_rgb(( i/len(_names), 1, 1 ))
		ax.errorbar(_coordsList[i][0], _coordsList[i][1]/np.max(np.absolute(_coordsList[i][1])), color=colour_, label=_names[i])
	ax.legend()
	plt.show()
	return 1

# resized the channel to fit it within a box of [width,height]=_sizeLims
def getResizedImage(_image,_sizeLims):
	#image_dims=( _image.shape[1], _image.shape[0] )# (width,height)
	size_mult=min([ _sizeLims[i]/_image.shape[i] for i in range(0,len(_sizeLims)) ])
	#size_mult = min([ _sizeLims[0]/image_dims[0], _sizeLims[1]/image_dims[1] ])
	outp_image_dims = tuple([int(i_id*size_mult) for i_id in _image.shape])
	return cv2.resize(_image, outp_image_dims[-1::-1])

# for testing purposes
def plotNumPixVsBrightness(_channel):
	printLog("start")
	allPixels=np.sort(np.concatenate(_channel))# in ascending order of brightness
	pixHisto,pixBins=np.histogram(allPixels, bins=1000)
	#
	plt.hist(allPixels, bins=1000)
	plt.xlabel("brightness")
	plt.ylabel("number of pixels")
	plt.show()
	return 1
