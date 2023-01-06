import sys
sys.path.append("c:\\users\\user\\miniconda3\\lib\\site-packages")

import cv2
import PIL
from datetime import datetime as dt
import openflexure_microscope_client as ofm_client

def main():
    microscope = ofm_client.find_first_microscope()
    captureAndSaveImage(microscope,"_img_.png")
    f=open("_img_complete_indicator_.txt","w")
    f.write("1")
    f.close()
    return 1

def captureAndSaveImage(_microscope,_name=None):
    if _name==None: _name=str(dt.timestamp(dt.now()))+".png"
    _microscope.capture_image().save(_name)
    return 1

main()