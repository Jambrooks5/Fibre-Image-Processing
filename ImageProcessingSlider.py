#Import the required library
import numpy as np
import PIL.Image
import tkinter as tk

from tkinter import *
from tkinter import ttk
from skimage.color import rgb2gray
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage import io
from PIL import Image,ImageTk


def get_masked_image(_mask_mult):
    image_array = np.array(opened_image) #convert image to numpy array #FORMAT SPECIFIC?
    outp_image_array = rgb2gray( image_array[...,0:3] ) # converts to greyscale
    
    #calculaste masking value for slider position and average brightness
    _mask_value = get_array_average(outp_image_array) * _mask_mult/30
    
    mask_array = outp_image_array <= _mask_value
    outp_image_array[mask_array] = 0# sets values <= _mask_value to 0
    mask_array = outp_image_array > _mask_value
    outp_image_array[mask_array] = 255# sets values > _mask_value to 255
    
    outp_image = PIL.Image.fromarray(outp_image_array)# FORMAT SPECIFIC
    return outp_image

#Find average brightness of orignal image 
def get_array_average(_array): 
    return np.sum(_array) / _array.size

#Define a Function to update the image with currrent mask slider value
def update_img(_mask_multiplier):
   mask_multiplier=float(_mask_multiplier)
   global masked_image
   masked_image = get_masked_image(mask_multiplier) #call masking function
   
   #Update GUI with new masked image
   global maskedImageTk
   maskedImageTk = ImageTk.PhotoImage( masked_image )
   label.configure(image=maskedImageTk)
   label.image=maskedImageTk
   #return masked_image

def stage2():
    print("Stage2")
    #openedImageRaw.show()
    opened_image = openedImageRaw
    update_img(slider.get())
    masked_image.show()

#Create an instance of tkinter frame
win= Tk()

screenWidth, screenHeight = win.winfo_screenwidth()/2, win.winfo_screenheight()

win.geometry('%dx%d+0+0' % (screenWidth,screenHeight))


#Define geometry of the window
#win.geometry("750x500")

#Load the image
opened_image=Image.open("uncleavedneedleburner.png")
openedImageRaw = Image.open("uncleavedneedleburner.png")
#openedImageRaw.show()
opened_image.thumbnail((500,500))

#Shows unmasked imaged before mask value changed
unmasked_image= ImageTk.PhotoImage(opened_image)
label= Label(win,image= unmasked_image)
label.pack()


#Create a slider to vary image masking value
slider = tk.Scale(win, from_=0, to=100, command=update_img, length=(500), orient=HORIZONTAL, label="Mask Threshold", tickinterval=20, resolution=0.5) 
slider.pack(pady=15)
win.bind("<Return>", update_img)

#Button to say when masking is complete
#complete = False
button = Button(win, text="Looks good", command=stage2)
button.pack()

win.mainloop()
    
    

    
    