from tkinter import *
from PIL import ImageTk,Image

def show_values():
    print (mask_slider.get())
    mainloop()

master = Toplevel()
image = Image.open('untapered.png')
image.thumbnail((500,500))
display = ImageTk.PhotoImage(image)
label = Label(master, image=display)

label.pack()

mask_slider = Scale(master, from_=0, to=100, orient=HORIZONTAL)
mask_slider.set(50)
mask_slider.pack()
Button(master, text='Apply', command=show_values()).pack()

mainloop()