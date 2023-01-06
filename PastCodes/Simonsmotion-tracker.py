# note: should lockdown wifi || connections while active, possibly excluding wifiSpeedLogger & sending video if necessary
# when the windows have opened, pressing 'p' will pause/unpause, and'q' will close the program; to allow for the closing procedure do not close through other methods unless necessary

####	SETUP

import cv2, time, os, sys
from datetime import datetime as dt
import numpy as np

# changes working directory to the file directory
currentDirPath=os.path.dirname(os.path.realpath(__file__))
os.chdir(currentDirPath)

####	PARAMETERS

prevFrameUpdateTime=0.2# the time between updates to the 'comparison frame' used for motion detection
prevFrameMotionMode=True# if True, updates the 'comparison frame' whenever motion is detected, instead of after time intervals ↑

blurKernelDims=(15,15)# the (x,y) dimensions of the Gaussian blur kernel used in getChannelChanges; helps remove noise; (1,1) applies no blur; must both be odd numbers
binaryThresholdLim=10# the minimum change in colour required to round up in the conversion to binary channel changes
minimumMotionDetectionSize=50# the minimum size a binary frame change has to be, in order to regester as motion in getMotionTracking

recordMode=1# if True, will record after any frames after motion is detected, and will combine them into a video when the program closes (the 'q' button)
afterMotionRecordingTime=0.2# how long it will record for after detecting motion; set to 0.0 to only record frames containing motion
videoFps=30# the fps of the video it generates; ~30 seems to be the camera fps
videoSmallestMode=False# if True the video will be the size of the smallest frame, else it will be the size of the largest instead; it should make no difference

motionDisplayMode=0# determines how motion is highlighted; ==0:squares; ==1:all contours; ==2:contours with area>minimumMotionDetectionSize
motionDisplayLineSize=2# determines the thickness of the motion highlighting lines
monochromeHighlightMode=0# if False, combines the motion highlighting lines, rather than pasting them over each other

####	MOTION TRACKING FUNCTIONS

def getChannelChanges( frame1In, frame2In, blurKernelDimsIn=(1,1) ):# returns the diffenence between 2 frames, split into [blue,green,red]
	frame1_split=cv2.split(frame1In)# splits the frame into 3 greyscale images AKA CHANNELS: (blue,green,red)
	frame2_split=cv2.split(frame2In)# ↑
	frame1_blur=[cv2.GaussianBlur(x,blurKernelDimsIn,0) for x in frame1_split]# performs a Gaussian blur on the channels
	frame2_blur=[cv2.GaussianBlur(x,blurKernelDimsIn,0) for x in frame2_split]# ↑
	out=[cv2.absdiff( frame1_blur[i], frame2_blur[i] ) for i in range(0,3)]# subtracts one frame from the other (frame2-frame1), one colour at a time
	return out

def getBinaryChannelChanges(channelChangesIn):# converts a set of channels into binary, ie if a pixel is brighter than binaryThresholdLim then it is output as max, else it is output as 0
	global binaryThresholdLim
	return [cv2.dilate( cv2.threshold(x, binaryThresholdLim, 255, cv2.THRESH_BINARY)[1] , None ) for x in channelChangesIn]

def getMotionTracking(frameIn,binaryChannelChangesIn):
	global minimumMotionDetectionSize
	channelsOut=cv2.split(frameIn)
	frameOut=cv2.merge(channelsOut)
	colour= (255,255,255) if not monochromeHighlightMode else (0,0,255)# note each channel is effectively greyscale, so maxing channel brightness will max that colour when merged
	motionDetected=False
	for i0 in range(0,3):# for each channel
		contours,heirarchy = cv2.findContours(binaryChannelChangesIn[i0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		targetImage= channelsOut[i0] if not monochromeHighlightMode else frameOut# determins whether the highlighting is done to each channel and then merged, or to the output frame, causing the highlights to be 'pasted' over each other, meaning they should be monochromatic
		for i1 in contours:
			if cv2.contourArea(i1) < minimumMotionDetectionSize:# if the contour isn't large enough to count as a detection
				continue
			motionDetected=True

			if motionDisplayMode==0:# rectangles
				(x,y,w,h) = cv2.boundingRect(i1)
				cv2.rectangle(targetImage, (x,y), (x+w,y+h), colour, motionDisplayLineSize)
			elif motionDisplayMode==2: cv2.drawContours(targetImage, i1, -1, colour, motionDisplayLineSize)# motion-detection triggering contours
		if motionDisplayMode==1 and motionDetected: cv2.drawContours(targetImage, contours, -1, colour, motionDisplayLineSize)# all contours

	if not monochromeHighlightMode: frameOut=cv2.merge(channelsOut)
	return frameOut, motionDetected

####	VIDEO GENERATION FUNCTIONS

def getImageDimsSet(_targetDirPath, smallestMode=False):
	frameFileNames=os.listdir(_targetDirPath)
	frameFileNames.sort()# sorts by value/alphabetically ascending
	frameDimsList=[]
	for i in frameFileNames:
		image=cv2.imread(_targetDirPath+'/'+i)
		frameDimsList.append( (len(image[0]), len(image)) )
	return set(frameDimsList)

def generateVideo(_targetDirPath,_videoName,_fps,smallestMode=False):# converts a directory of saved frames to a video and saves it
	frameFileNames=os.listdir(_targetDirPath)
	frameFileNames.sort()# sorts by value/alphabetically ascending
	# ↓ resizing
	print("generateVideo: getting image dimensions")
	imageDimsSet=getImageDimsSet(_targetDirPath)
	if smallestMode:
		def replaceCheck(oldIn,newIn): return newIn<oldIn
	else:
		def replaceCheck(oldIn,newIn): return newIn>oldIn
	finalDims=list(list(imageDimsSet)[0])# imageDimsSet is a set of sets
	for i in imageDimsSet:
		if replaceCheck(finalDims[0],i[0]): finalDims[0]=i[0]
		if replaceCheck(finalDims[1],i[1]): finalDims[1]=i[1]
	# ↓ generation
	print("generateVideo: generating video")
	videoFile=cv2.VideoWriter(_videoName, cv2.VideoWriter_fourcc(*'DIVX'), _fps, finalDims)
	for i in frameFileNames:
		image=cv2.imread(_targetDirPath+'/'+i)
		image=cv2.resize(image, finalDims, interpolation=cv2.INTER_CUBIC)
		videoFile.write(image)
	videoFile.release()
	print("generateVideo: ended")
	return 1

####	MISCELLANEOUS FUNCTIONS

# note this ↓ only works if vid & ret,frame are already defined; also prevTime=0, and motionDetected=True  must be set before the loop
def updateFrames():# updates the video frames
	global vid, prevFrameUpdateTime, prevFrameMotionMode, prevTime, time, prevFrame, ret,frame, motionDetected
	time=dt.timestamp(dt.now())
	# ↓ if (prevFrame updates aren't auto, and the prevFrameUpdateTime has elapsed since the last 'previous frame' was taken) or (prevFrame updates are on auto, and a frame change has been detected): replace prevFrame with the current frame
	if (not prevFrameMotionMode and time-prevTime >= prevFrameUpdateTime) or (prevFrameMotionMode and motionDetected):
		prevTime=time
		prevFrame=frame
	ret,frame = vid.read()# capture a video frame
	return 1

def deleteDir(dirPathIn):
	containedFileNames=os.listdir(dirPathIn)
	for i in containedFileNames:
		os.remove(dirPathIn+'/'+i)
		print('deleted:',dirPathIn+'/'+i)
	os.rmdir(dirPathIn)
	print('deleted:',dirPathIn)
	return 1

def replace_chars(string_in, old_chars, new_chars):#note new chars can be strings instead
	string_list=list(string_in)#split string into list for editing
	for string_i in range(0,len(string_list)):#for each character in string_list...
		for chars_i in range(0,len(old_chars)):#...replace it if its equal to anything in old_chars
			if string_list[string_i]==old_chars[chars_i]: string_list[string_i]=new_chars[chars_i]
	return "".join(string_list)#turn list back into string

def getFilenameCompatableTime(_time): return replace_chars(str(_time), [' ',':'], ['__','-'])# returns a string that can be used as a file name from the output of eg datetime.now()

####	MAIN

startTime=getFilenameCompatableTime(dt.now())
savePath=os.path.dirname(os.path.realpath(__file__))+'/motion-tracker-vid-'+startTime
if recordMode and not os.path.exists(savePath): 
	os.mkdir(savePath)
	print('created:',savePath)

vid = cv2.VideoCapture(0)# defines a video capture object
prevTime=0
ret,frame = vid.read()
motionDetected=True# necessary if prevFrameMotionMode==True, see updateFrames()
while True:
	# ↓ handles the frame capture and processing
	updateFrames()
	unbluredFrameChange=cv2.merge(getChannelChanges( prevFrame, frame, (1,1) ))
	channelChanges=getChannelChanges(prevFrame,frame,blurKernelDims)
	frameChange=cv2.merge(channelChanges)
	binaryChannelChanges=getBinaryChannelChanges(channelChanges)
	binaryFrameChange=cv2.merge(binaryChannelChanges)
	motionTrackingFrame,motionDetected=getMotionTracking(frame,binaryChannelChanges)
	# ↓ handles the image display
	cv2.imshow('unblured frame change',unbluredFrameChange)# displays the required frames
	cv2.imshow('frame change',frameChange)# ↑
	cv2.imshow('binary frame change',binaryFrameChange)# ↑
	cv2.imshow('motion tracking',motionTrackingFrame)# ↑
	# ↓ handles the recording
	if recordMode and time-prevTime<=afterMotionRecordingTime:
		compatableTime=getFilenameCompatableTime(dt.fromtimestamp(time))
		status=cv2.imwrite(savePath+'/img-'+compatableTime+'.png',frame)
		print(status,'saved:',savePath+'/img-'+compatableTime+'.png')
	# ↓ handles the user input
	pressedKey=cv2.waitKey(1)&0xFF# the &0xFF excludes the latter part of the byte, which can sometimes change eg if numlock is on
	if pressedKey == ord('p'):# the 'p' button is set as the pause button
		vid.release()# so that it also stops its connection to the camera while paused
		print('paused')
		while True:# the pause waiting loop
			pressedKey=cv2.waitKey(1)&0xFF
			if pressedKey == ord('p'):
				print('unpaused')
				break
			elif pressedKey == ord('q'): break# if q is pressed, it will carry over to the closing sequence after this break
		vid = cv2.VideoCapture(0)# to re-connect to the camera
	if pressedKey == ord('q'):# the 'q' button is set as the quitting button
		print('closing process: beginning, do not interrupt untill complete')
		break
# ↓ is the closing process
endTime=getFilenameCompatableTime(dt.now())
vid.release()# After the loop release the cap object.
cv2.destroyAllWindows()# Destroy all the windows.
if recordMode: 
	generateVideo(savePath,'motion-tracker-vid.avi',videoFps,videoSmallestMode)# 'motion-tracker-vid-'+startTime+'.avi'
	#deleteDir(savePath)
infoFile=open('motion-tracker-vid-info-'+startTime+'.txt','w+')
infoFile.write('start: '+startTime+'\nend: '+endTime)
infoFile.close()

print('closing process: complete\nended')

