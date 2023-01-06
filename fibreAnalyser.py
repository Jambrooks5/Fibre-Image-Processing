import generalFunctions as gf
#import fibreFinder as ff
import numpy as np
from scipy import ndimage
import cv2


		# PARAMETERS


#imageFileName="untapered.png"
#imageFileName="cane1.jpg"
imageFileName="cane2.jpg"
#imageFileName="cane3.jpg"
#imageFileName="cane4.jpg"
#imageFileName="stellaris.jpg"

displayLims=[800,1500]# ←[y_limit,x_limit]
numPixProcessingLim=2**16# 0 to disable.

blurKFractionalWidth=1/100# blur kernel size as a fraction of the channel size


		# MAIN


def main():
	channel=cv2.imread(imageFileName,0)/255
	blurKWidth=int(blurKFractionalWidth*(channel.size**.5)/2)*2+1# The surrounding calclations ensure that it is an odd integer.
	bluredChannel=cv2.GaussianBlur(channel,(blurKWidth,blurKWidth),0)
	resizedChannel=gf.getSizeLimited(bluredChannel,numPixProcessingLim)
	edgeChannel=getAltSobelEdgeChnl(resizedChannel)
	#
	mirrorChannel=gf.getNormalised(getMirrorSymmetryChannel(edgeChannel))
	mirrorCenter=gf.getArrayMaxIndex(mirrorChannel)
	gf.printLog("mirrorCenter="+str(mirrorCenter))
	flipChannel=gf.getNormalised(get180FlipSymmetryChannel(edgeChannel))
	flipCenter=gf.getArrayMaxIndex(flipChannel)
	gf.printLog("flipCenter="+str(flipCenter))
	#
	angleRadiusChannel=getAngleRadiusChannel(edgeChannel,_center=mirrorCenter,_rStretchMult=15)
	#
	sumVsRadius=getFunctionOfPerimiterValuesVsRadius(edgeChannel,mirrorCenter,np.sum)
	aveVsRadius=getFunctionOfPerimiterValuesVsRadius(edgeChannel,mirrorCenter,np.average)
	maxVsRadius=getFunctionOfPerimiterValuesVsRadius(edgeChannel,mirrorCenter,np.max)
	#fractionVsRadius=getFunctionOfPerimiterValuesVsRadius(edgeChannel,mirrorCenter,lambda x: np.max())
	#
	gf.showImages( ["channel","bluredChannel","resizedChannel","edgeChannel","mirrorChannel","flipChannel","angleRadiusChannel"], [channel,bluredChannel,resizedChannel,edgeChannel,mirrorChannel,flipChannel,angleRadiusChannel], displayLims )
	gf.plotDataSequences( ["sumVsRadius","aveVsRadius","maxVsRadius"], [sumVsRadius,aveVsRadius,maxVsRadius] )
	cv2.destroyAllWindows()
	return 1


		# EDGE CHANNEL


# each outp pixel is the input brightness on one side devided by the input brightness on the other side
def getFractionalEdgeChnl(_channel):
	gf.printLog("start")
	_kernel=np.array([[1,2,1],[2,0,2],[1,2,1]])
	outp=np.zeros(_channel.shape, dtype=float)
	kernel=_kernel/np.sum(np.absolute(_kernel))
	for iY in range(0,_channel.shape[0]-kernel.shape[0]):
		gf.printUpdatePercentage(iY,_channel.shape[0]-kernel.shape[0]-1)
		for iX in range(0,_channel.shape[1]-kernel.shape[1]):# note this leaves the outer edge outp pixels black
			thisPixPos=thisPixPos=(iY+int(kernel.shape[1]/2) ,iX+int(kernel.shape[0]/2))
			thisPix=0
			for iKY in range(0,kernel.shape[0]):
				for iKX in range(0,kernel.shape[1]):
					thisPix+=kernel[iKY][iKX]*( _channel[iY+iKY][iX+iKX]/_channel[iY+kernel.shape[0]-iKY-1][iX+kernel.shape[1]-iKX-1] - 1 )
			outp[thisPixPos[0]][thisPixPos[1]]=thisPix
	gf.printLog("end")
	return gf.getNormalised(outp)

#↓ performs a modified (fractional) sobel operation, ie gradient/brightness; _channel must be greyscale, with float elements
def getAlteredSobel(_channel,_kernel):
	gf.printLog("start")
	outp=np.zeros(_channel.shape, dtype=float)# ndarray.shape is the dimensions of the array, ie =(xLen,yLen)
	kernel=_kernel/np.sum(np.absolute(_kernel))# normalises the kernel to give it magnitude=1, so it never outputs a value outside the pixel brightness range, ie (0,255) or (0.0,1.0)
	for iY in range(0,_channel.shape[0]-kernel.shape[0]):
		gf.printUpdatePercentage(iY,_channel.shape[0]-kernel.shape[0]-1)
		for iX in range(0,_channel.shape[1]-kernel.shape[1]):# note this leaves the outer edge outp pixels black
			thisPixPos=(iY+int(kernel.shape[1]/2) ,iX+int(kernel.shape[0]/2))# pix=>pixel
			thisPix=0
			thisSectionBrightness=0
			thisSection=_channel[ iY:iY+kernel.shape[1], iX:iX+kernel.shape[0] ]
			thisPix=abs(np.sum(np.multiply(thisSection,kernel) ) )
				# ↓ temporary, we must choose 1
			#thisSectionBrightness=_channel[thisPixPos[0]][thisPixPos[1]]
			#thisSectionBrightness=thisSection.max()
			#thisSectionBrightness=np.sum(thisSection)/thisSection.size
			thisSectionBrightness=thisSection.min()
			# ↓ converts to ~ the change in the fraction of the brightness
			thisPix/=thisSectionBrightness
			outp[thisPixPos[0]][thisPixPos[1]]=thisPix
	gf.printLog("end")
	return outp

def getAltSobelEdgeChnl(_channel):
	sobelXKernal=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
	sobelYKernal=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	altGradXChannel=getAlteredSobel(_channel,sobelXKernal)
	altGradYChannel=getAlteredSobel(_channel,sobelYKernal)
	outp=np.sqrt(np.add( np.power(altGradXChannel,2), np.power(altGradYChannel,2) ))# (X^2+Y^2)^1/2; remove sqrt? helps to seperate walls from noise?
	return outp


		# FINDING THE CENTER


# Each pixel of the gradient is a vector. Make a line across the image for each of these pixels (with direction and amplitude from the vector) and sum them.
# Hypothesis: this sum will be highest at the center?
	# May be true for some fibres, but definitely not all. Consider counter example of the something like the rotorFlex component.

# Enter the fibre symmetry (or find it from the Fourier transform fringes) and find the location where that rotational symmetry factor is highest.
	# Note that even if the symmetry is broken, the original image and it's Fourier transform should be broken in the same way, so the method may still work.

# Fourier transform phases? The repeating pattern is likely centered on the center of the fibre.
	# For every possible direction, the plane waves must create a series of peaks which will be symmetrical around the center of the fibre?

# If the outer/inner diameter has been found: sum together the pixels of that circle/circles together for each possible center pixel

# Overlay the Fourier image over the original, centered at different points, accounting for the conversion between real space & wavevector space.
	# The Fourier transform fringes will allign with the capillaries.
	# The low wavevector peaks should correspond to the outer\inner diameter? Yes but would be difficult to use without reconstruction.

# Is it possible to perform a phase shift of the Fourier image in order to shift the symetrical structure to the center of the reconstructed real-space image?
	# Create a vaugue reconstruction of the original image, but centered, and overlay that on the original over all positions and find where the overlap is maximum?
# Say we start with a centrally symmetric real image; what is the difference between that and an image with it's center of symmetry elsewhere?
	#  It probably helps to consider the repeating version of the reconstruction.
# We may simply be able to extract the center of symmetry from the Fourier phases directly, rather than trying to fit a centralised approximation of the image to the original (which would be effective but slow).

# Find the dominant radial sinusoids in the Fourier image, use them to find the dominant radii in the original image, use those to collapse the r dimension of the Hough transform.

# Start at each center point, scan outwards and for each radius see how similar the points on the corresponding circle are to each other (IE the lowest variance or standev?)
	# This is probably related to the Hough Transform of the image, but collapsed in the r dimension

# We could start with an approximate (using EG the weighted average position of the pixels) then use a strategy that will tend toward the 'true' center
# The deffinition of the 'true' center is important here
	# The center of the largest circle?

#
def getFibreCenter(_channel):
	mirrorSymmetryChannel=getMirrorSymmetryChannel(_channel)
	initialOutpGuess=gf.getArrayMaxIndex(mirrorSymmetryChannel)
	return 0


		# FINDING THE OUTER/INNER DIAMETER


# If the center has been found: Simply add all the pixel values at each radius and find the peak/peaks
	# There may be false peaks due to capillaries, the outer\inner diameters can be isolated by noting that the average\sum of the radial pixels should be low between the 2.
		# Should account for the fact that the outer diameter may not always be in view.

# Search for low frequency circular peaks in the fourier image

# Use peaks in getFunctionOfPerimiterValuesVsRadius

# TODO: ↓.
# Take a radius step, and for each theta check the difference between the pixels. The more that have changed at once, the more likely it is to be the border of a circle.
	# Maybe fraction instead of sum, so that it measures the 'fraction' of the circle.
	# Could also be used to hone the center?


		# FINDING THE DIAMETERS & CENTER TOGETHER


# could possible find the outer/inner diameter and the center at the same time by:
	# for each possible center pixel:
		# sum_over_all_pixels( multiply the pixels at each radius by the magnitude of the fourier transform (summed in a circle) at the corresponding wavevector magnitude )
	# see which center pixels produce the highest total, these whould be the centers of circle-like things
		# perhaps skew this towards larger circles ie lower frequencies? without finding circles where there are none of course.
# This is essentially an alternate version of the circle finding algorithm that James has been using (which is rather clunky, which doesn't bode well)


		# FINDING THE CORE / CENTRAL VOID DIAMETER


# Use the maximum brightness at each radius, find where it first increases to a large value (an appreciable fraction of the maximum value) and stays there for longer than noise.
# Alternatively, just look for the first change in character when scanning out from the center.
def getHollowCoreRadius(_channel,_center):
	return 0


		# QUANTISING THE ROTATIONAL SYMMETRY


# If the center is found:
	# rotate around the center for each possible number of capillaries (or possibly just prime numbers & then calculate; no, wouldn't account for products of squares)
	# Also, should use a multi_image_overlap function, so we can rotate the image into each of the symmetryNum rotations and just get the overlap of the whole lot (would need a different overlap method to the minimum
		# perhaps the product? (no, would favour lower symmetryNum)
		# perhaps the average? (maybe, but a single cappilary out of place wouldn't make a huge difference)
		# perhaps a product equivalent of the average such as Product(i=0...n-1){imageRotation_i}**(1/n)? (give it a try, but it sounds promising)


		# FINDING THE NUMBER & POSITIONS OF ALL CAPILLARIES, REGARDLESS OF FORMATION OR SHAPE


# Some measure of how 'likely' each pixel is to be the center of a hole region.
# Or the likelyhood of simply being within a hole region, which may be more versatile.


		# FOURIER FUNCTIONS


#↓ performs Fourier transform, and centers the origin
def getFourierDisplayChannels(_channel):
	gf.printLog("start")
	fourier = np.fft.fft2(_channel)
	centeredFourier = np.fft.fftshift(fourier)# shifts the origin (|k|=0) of the fourier spectrum to the center of the image
	magChannel = np.abs(centeredFourier.copy())
	magChannel=gf.getNormalised(np.log(magChannel))
	#magChannel*=50# TEMP. Replace with fraction of pixels displayed above 1?
	phaseChannel=np.absolute(np.angle(centeredFourier)-np.pi)/np.pi
	reChannel=gf.getNormalised(np.log( np.abs(np.real(centeredFourier)) ))
	reChannel[reChannel==-np.inf]=0
	imChannel=gf.getNormalised(np.log( np.abs(np.imag(centeredFourier)) ))
	imChannel[imChannel==-np.inf]=0
	gf.printLog("end")
	return [magChannel,phaseChannel,reChannel,imChannel]

#
def getFourierDephasedChannel(_channel): return np.absolute(np.fft.fftshift(np.fft.ifft2( np.absolute(np.fft.fft2(_channel)) )))


		# SYMMETRY FUNCTIONS	Note: the maximum of any of the symmetry channels can be used as a decent approximation for the center of the fibre.


# _center is the index of the position along the chosen axis around which the reflection occurs.
def getChannelYSymmetryFactor(_channel,_center=None):
	if _center!=None:
		cropRadius=min([_center,_channel.shape[0]-(_center+1)])# [-,-,-,c,-,-] ==> [0-1-2-3c4-5-6]
		croppedChannel=_channel[ _center-cropRadius:_center+1+cropRadius ]
	else: croppedChannel=_channel
	outp=getOverlapFactor(croppedChannel,cv2.flip(croppedChannel,0))
	return outp

# _center is the index of the position along the chosen axis around which the reflection occurs.
def getChannelXSymmetryFactor(_channel,_center=None):
	if _center!=None:
		cropRadius=min([_center,_channel.shape[1]-(_center+1)])# [-,-,-,c,-,-] ==> [0-1-2-3c4-5-6]
		croppedChannel=_channel[ :, _center-cropRadius:_center+1+cropRadius ]
	else: croppedChannel=_channel
	outp=getOverlapFactor(croppedChannel,cv2.flip(croppedChannel,1))
	return outp

# NOTE the edge pixels will all be 0.
# Fast, can handle 2**20 pixels in under 2 seconds.
# t~O(p**1.5)
def getMirrorSymmetryChannel(_channel):
	gf.printLog("start")
	ySymArr=np.array([ getChannelYSymmetryFactor(_channel,_center=iY)*gf.printUpdatePercentage(.5*(iY-1),_channel.shape[0]-3) for iY in range(1,_channel.shape[0]-1) ])
	xSymArr=np.array([ getChannelXSymmetryFactor(_channel,_center=iX)*gf.printUpdatePercentage(.5*(iX-1)+.5*(_channel.shape[1]-3),_channel.shape[1]-3) for iX in range(1,_channel.shape[1]-1) ])
	outp=np.zeros(_channel.shape)
	for iY in range(1,_channel.shape[0]-1):
		#gf.printUpdatePercentage(iY-1,_channel.shape[0]-3)
		for iX in range(1,_channel.shape[1]-1):
			outp[iY][iX]=( xSymArr[iX-1] + ySymArr[iY-1] )# TODO: what is more mistake-resistant: (xs**2+ys**2)**.5, xs+ys, xs*ys, min(xs,ys)
	gf.printLog("end")
	return outp

#
def get180FlipSymmetryFactor(_channel,_center=()):
	center=gf.getArrayCenterIndex(_channel) if _center==() else _center
	croppedChannel=getCroppedAroundCenter(_channel,center,False)
	rotatedChannel=cv2.flip(croppedChannel,-1)
	outp=getOverlapFactor(croppedChannel,rotatedChannel)
	return outp

# Slower, but can still run reasonably at 2**16 pixels.
# t~O(p**2)
def get180FlipSymmetryChannel(_channel):
	gf.printLog("start")
	outp=np.zeros(_channel.shape)
	for iY in range(1,_channel.shape[0]-1):
		gf.printUpdatePercentage(iY-1,_channel.shape[0]-3)
		for iX in range(1,_channel.shape[1]-1):
			outp[iY][iX]=get180FlipSymmetryFactor(_channel,_center=[iY,iX])# t1~O(p)?
	gf.printLog("end")
	return outp

#
def getRotationalSymmetryFactor(_channel,_symmetryNum,_center=[]):
	outp=0
	for i in range(1,_symmetryNum):
		rotationMatrix=cv2.getRotationMatrix2D(_center[-1::-1], i*360/_symmetryNum, 1.)
		rotatedChannel=cv2.warpAffine( _channel, rotationMatrix, _channel.shape[-1::-1] )
		#rotatedChannel=ndimage.rotate( _channel, i*360/_symmetryNum, reshape=False )
		outp+=getOverlapFactor(_channel,rotatedChannel)/(_symmetryNum-1)
	return outp

# Slow, not really usable for more than 2**14 pixels.
# NOTE the edge pixels will all be 0.
def getRotationSymmetryChannel(_channel,_symmetryNum):
	gf.printLog("start")
	outp=np.zeros(_channel.shape)
	for iX in range(1,_channel.shape[0]-1):
		gf.printUpdatePercentage(iX-1,_channel.shape[0]-3)
		for iY in range(1,_channel.shape[1]-1):
			outp[iX][iY]=getRotationalSymmetryFactor(_channel,_symmetryNum,_center=[iX,iY])# t1~O(p)?
	gf.printLog("end")
	return outp


		# OTHER IMAGE ANALYSIS FUNCTIONS


# Input images must have the same dimensions and have elements in the range 0.0 to 1.0.
def getOverlapImg(_img1,_img2):
	#return np.ones(_img1.shape,dtype=float)-np.absolute(_img1-_img2)# No
	#return np.multiply(_img1,_img2)
	return np.minimum(_img1,_img2)

#
def getOverlapFactor(_img1,_img2): return np.sum(getOverlapImg(_img1,_img2))#/_img1.size


		# IMAGE CROPPING FUNCTIONS


# t~O(p)?
def getCircleMask(_shape,_radius=0):
	radius= min([ int((i-1)/2) for i in _shape ]) if _radius==0 else _radius
	ogrid_=np.ogrid[0:_shape[0], 0:_shape[1]]
	outp=(ogrid_[0]-_shape[0]/2)**2 + (ogrid_[1]-_shape[1]/2)**2 < radius**2
	return outp

# The smallest of the distances from the center to each edge.
def getMinDistToEdge(_channelShape,_center):
	return min([ _center[0], _channelShape[0]-(_center[0]+1), _center[1], _channelShape[1]-(_center[1]+1) ])# [-,-,-,c,-,-] ==> [0-1-2-3c4-5-6]

# As with getMinDistToEdge but with corners, and maximum.
def getMaxDistToEdge(_channelShape,_center):
	distancesToCorners=[0,0,0,0]# [tl,tr,bl,br]
	distancesToCorners[0]=( (_center[0])**2 + (_center[1])**2 )**.5
	distancesToCorners[1]=( (_center[0])**2 + (_channelShape[1]-_center[1]-1)**2 )**.5
	distancesToCorners[2]=( (_channelShape[0]-_center[0]-1)**2 + (_center[1])**2 )**.5
	distancesToCorners[3]=( (_channelShape[0]-_center[0]-1)**2 + (_channelShape[1]-_center[1]-1)**2 )**.5
	return max(distancesToCorners)

#
def getCroppedAroundCenter(_channel,_center,_enforceSquareMode=False):
	if _enforceSquareMode:
		cropRadius=getMinDistToEdge(_channel.shape,_center)
		cropRadii=[cropRadius,cropRadius]
	else:
		cropRadii=[ min([ _center[0],_channel.shape[0]-(_center[0]+1)]), min([_center[1],_channel.shape[1]-(_center[1]+1)]) ]
	cropBounds=[ [_center[0]-cropRadii[0],_center[0]+1+cropRadii[0]], [_center[1]-cropRadii[1],_center[1]+1+cropRadii[1]] ]# [ [x0,x1], [y0,y1] ]
	return _channel.copy()[cropBounds[0][0]:cropBounds[0][1],cropBounds[1][0]:cropBounds[1][1]]


		# RADIAL ANALYSIS FUNCTIONS


# Returns the indicies on the perimiter of a circle with radius _r and [y,x] position _center.
def getPerimiterIndicies(_r,_center=(0,0)):
	numThetaSteps=int(_r*np.pi/2)+1# The number of points taken over one quarter of the circle. Broken into quarters to retain symmetry for low radii.
	outp=np.zeros((numThetaSteps*4,2),dtype=np.int_)
	for iQuarter in range(0,4):
		for i in range(0,numThetaSteps):
			th=.5*np.pi*(iQuarter + i/numThetaSteps)
			outp[int(iQuarter*numThetaSteps)+i,0]=round(_r*np.sin(th)+_center[0])# y
			outp[int(iQuarter*numThetaSteps)+i,1]=round(_r*np.cos(th)+_center[1])# x
	return outp

#
def getPerimiterValues(_channel,_r,_center):
	return np.array([ ( _channel[i[0]][i[1]] if (i[0]<_channel.shape[0] and i[0]>-1 and i[1]<_channel.shape[1] and i[1]>-1) else 0 ) for i in getPerimiterIndicies(_r,_center) ])

# The default _func will simply return each list of perimiter pixels for each radius, but a function of each list of perimiter pixels can be passes instead.
def getFunctionOfPerimiterValuesVsRadius(_channel,_center,_func=(lambda x: x)):
	rMax=int(getMaxDistToEdge(_channel.shape,_center))+1
	outp=np.zeros((2,rMax),dtype=_channel.dtype)
	for iR in range(0,rMax):
		r=iR
		outp[0][iR]=r
		outp[1][iR]=_func( getPerimiterValues(_channel,r,_center) )
	return outp# ←[ r values , f(perimiter) values ]

#
def getAngleRadiusChannel(_channel,_center=(),_rStretchMult=1):
	gf.printLog("start")
	center=gf.getArrayCenterIndex(_channel) if _center==() else _center
	rMax=int(getMaxDistToEdge(_channel.shape,center))
	numThetaSteps=int(rMax*2*np.pi)+1
	outp=np.zeros( (rMax*_rStretchMult,numThetaSteps), dtype=_channel.dtype )
	for iR in range(0,rMax*_rStretchMult):
		r=iR/_rStretchMult
		gf.printUpdatePercentage(iR,rMax*_rStretchMult-1)
		for iTh in range(0,numThetaSteps):
			th=2*np.pi*iTh/numThetaSteps
			pixIndex=( round(r*np.sin(th)+_center[0]), round(r*np.cos(th)+_center[1]) )
			outp[-iR-1][iTh]=_channel[ pixIndex ] if ( pixIndex[0]<_channel.shape[0] and pixIndex[0]>-1 and pixIndex[1]<_channel.shape[1] and pixIndex[1]>-1 ) else 0
	gf.printLog("end")
	return outp


		# RUN


if __name__=="__main__": main()

