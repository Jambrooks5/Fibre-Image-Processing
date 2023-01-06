def get_masked_image(_image, _mask_value):
    image_array = np.array(_image)
    outp_image_array = rgb2gray( image_array[...,0:3] )# converts to greyscale
    
    mask_array = outp_image_array <= _mask_value
    outp_image_array[mask_array] = 0# sets values <= _mask_value to 0
    mask_array = outp_image_array > _mask_value
    outp_image_array[mask_array] = 255# sets values > _mask_value to 255
    
    outp_image = PIL.Image.fromarray(outp_image_array)
    return outp_image
    



master = Toplevel()


#add brightness mask slider and apply button
mask_slider = Scale(master, from_=0, to=100, command=show_mask, orient = HORIZONTAL)
mask_slider.set(50)
mask_slider.pack()

image.thumbnail((500,500)) #resize image w/ fixed aspect ratio
display = ImageTk.PhotoImage(image)
label = Label(master, image=display)
label.pack()

shown = False
good_mask = False

#Button(master, text='Apply', command=show_mask(mask_slider.get())).pack(side='bottom')


mainloop()

#while (good_mask==False):
    
    #Button(master, text='Apply', command=(good_mask==True)).pack(side="bottom")
 #   print(good_mask)
  #  shown = show_mask(image, mask_slider.get(), master)
    
    