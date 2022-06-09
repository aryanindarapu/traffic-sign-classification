import tkinter as tk
from tkinter import *
from tkinter import filedialog
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array
from PIL import ImageTk, Image, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt

classes = { 
  1:'Speed limit (20km/h)',
  2:'Speed limit (30km/h)',
  3:'Speed limit (50km/h)',
  4:'Speed limit (60km/h)',
  5:'Speed limit (70km/h)',
  6:'Speed limit (80km/h)',
  7:'End of speed limit (80km/h)',
  8:'Speed limit (100km/h)',
  9:'Speed limit (120km/h)',
  10:'No passing',
  11:'No passing veh over 3.5 tons',
  12:'Right-of-way at intersection',
  13:'Priority road',
  14:'Yield',
  15:'Stop',
  16:'No vehicles',
  17:'Vehicle > 3.5 tons prohibited',
  18:'No entry',
  19:'General caution',
  20:'Dangerous curve left',
  21:'Dangerous curve right',
  22:'Double curve',
  23:'Bumpy road',
  24:'Slippery road',
  25:'Road narrows on the right',
  26:'Road work',
  27:'Traffic signals',
  28:'Pedestrians',
  29:'Children crossing',
  30:'Bicycles crossing',
  31:'Beware of ice/snow',
  32:'Wild animals crossing',
  33:'End speed + passing limits',
  34:'Turn right ahead',
  35:'Turn left ahead',
  36:'Ahead only',
  37:'Go straight or right',
  38:'Go straight or left',
  39:'Keep right',
  40:'Keep left',
  41:'Roundabout mandatory',
  42:'End of no passing',
  43:'End no passing vehicle with a weight greater than 3.5 tons' 
}

def preprocessing(img):
  grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  eqimg = cv2.equalizeHist(grayimg)
  eqimg = eqimg / 255
  return eqimg

window = tk.Tk()
window.geometry('800x600')
window.title('Traffic Sign Classification')
window.configure(background='#CDCDCD')
label = Label(window, background='#CDCDCD', font=('dubai', 16, 'bold'))
image = Label(window)
var = IntVar()

def classify(file_path):
  global label_packed
  label.configure(text='Classifying...')
  img = load_img(file_path, target_size=(32, 32))
  img_array = img_to_array(img, dtype='uint8')

  if var.get() == 1: 
    img_array = preprocessing(img_array)
    model = keras.models.load_model("./grayscale_models/with_padding.keras")
  else: model = keras.models.load_model("./rgb_models/with_relu_adam.keras")

  img_array = np.expand_dims(img_array, axis=0)
  pred = model.predict(img_array)
  sign = np.argmax(pred[0])
  print(sign)
  label.configure(foreground='#011638', text=classes[sign+1])
  
def show_classify_button(file_path):
  classify_b = Button(window, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
  classify_b.configure(background='#364156', foreground='white', font=('dubai', 14, 'bold'))
  classify_b.place(relx=0.79, rely=0.46)

def upload_image():
  try:
    file_path = filedialog.askopenfilename()
    uploaded = Image.open(file_path)
    if var.get() == 1: uploaded = ImageOps.grayscale(uploaded)
    uploaded.thumbnail(((window.winfo_width()), (window.winfo_height())))
    resized_image= uploaded.resize((150,150), Image.LANCZOS)
    im = ImageTk.PhotoImage(resized_image)
    image.configure(image=im)
    image.image = im
    label.configure(text='Image Ready To Be Classified')
    show_classify_button(file_path)
  except:
    pass

R1 = Radiobutton(window, text="RGB Model", variable=var, value=0)
R1.pack( anchor = W )
R2 = Radiobutton(window, text="Grayscale Model", variable=var, value=1)
R2.pack( anchor = W )

upload=Button(window, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('dubai', 14, 'bold'))
upload.pack(side=BOTTOM, pady=50)
image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(window, text="Classifying the German Traffic Sign", pady=10, font=('dubai', 24))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
window.mainloop()