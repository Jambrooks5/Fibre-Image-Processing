import os
import cv2
import FocusStack
"""

    Focus stack driver program

    This program looks for a series of files of type .jpg, .jpeg, or .png
    in a subdirectory "input" and then merges them together using the
    FocusStack module.  The output is put in the file merged.png


    Author:     Charles McGuinness (charles@mcguinness.us)
    Copyright:  Copyright 2015 Charles McGuinness
    License:    Apache License 2.0

"""

def stackHDRs(image_files):
    print("StackHDRs")
    focusimages = []
    for img in image_files:
        print ("Reading in file {}".format(img))
        focusimages.append(cv2.imread("input/{}".format(img)))

    merged = FocusStack.focus_stack(focusimages)
    cv2.imwrite("merged.png", merged)
    return merged

def stackStart():
    print("StartStack")
    #if __name__ == "__stackMain__":
    image_files = sorted(os.listdir("input"))
    for img in image_files:
        if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
            image_files.remove(img)


    finishedStack = stackHDRs(image_files)
    print ("Stacking done")
    
    return finishedStack
        

