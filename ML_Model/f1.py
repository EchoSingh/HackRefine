import tkinter as tk
from tkinter import filedialog
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os

import cv2
from tkinter import *
from PIL import ImageTk, Image
import _tkinter # with underscore, and lowercase 't'

win=tk.Tk()

def b1_click():
    global path2
    try:
        json_file = open('model1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model1.h5")
        print("Loaded model from disk")
        label=["Apple___Apple_scab suggest Pest-Captan,Chlorothalonil,Copper (Organic","Apple___Black_rot suggest to Pest-Avoid replanting into the same area that contained a previously infected plant.","Apple___Cedar_apple_rust suggest to Pest-Organicdews Neemoil,cinnaprot-proteco,Superkukulus-100ml","Apple___Healthy",
               "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot suggest to Pest-Copper oxychloride","Corn_(maize)___Common_rust_ suggest to Pest-sulfur or copper-based fungicides",
               "Corn_(maize)___Healthy","Corn_(maize)___Northern_Leaf_Blight suggest to Pest-Mancozeb,chlorothalonil 2","Grape___Black_rot suggest to Pest-Eco Oil,Eco Neem,Eco Traps",
               "Grape___Esca_(Black_Measles)","Grape___Healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
               "Potato___Early_blight","Potato___Healthy","Potato___Late_blight use Pest-Abtec Bio Neem Plant Pesticide-Neem Oil(Azadirachtin)-Control Aphids","Tomato___Bacterial_spot",
               "Tomato___Early_blight","Tomato___Healthy","Tomato___Late_blight","Tomato___Leaf_Mold use Pest-Agro Plus AM003_1 Pesticide",
               "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot use pest-agro pure water",
               "Tomato___Tomato_Yellow_Leaf_Curl_Virus use pest-agro fertilize to keep healthy","Tomato___Tomato_mosaic_virus use Pest-Agro Plus AM003_1 Pesticide"]

        
        #lbl2=tk.Label(win,image=img)
        
        #lbl2.pack(side = "bottom", fill = "both", expand = "yes")
        #img1=('F:/py/leaf_disease_final( COMPLETE )/1.jpg')


        #lbl2=tk.Label(win,image=img1)
        #lbl2.pack(side = "bottom", fill = "both", expand = "yes")
        #loading image 
        path2=filedialog.askopenfilename()
        print(path2)
        

        #img = ImageTk.PhotoImage(Image.open(path2))
        
        #lbl2=tk.Label(win,image=img)
        #lbl2.pack(side = "bottom", fill = "both", expand = "yes")

        #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
        #panel = tk.Label(win, image = img)
        #panel.pack( fill = "both", expand = "yes")
        #imr=cv2.imread(path2)
        #a=cv2.imshow(imr)
        #print(imr)
        test_image = image.load_img(path2, target_size = (128, 128))        
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        #print(result)
        #print(result)
        fresult=np.max(result)
        label2=label[result.argmax()]
        #print(label2)
        #lb2.configure(image=img)
        #lbl2.image=img
        lbl.configure(text=label2)
         
        
        #lbl2(ent.config(state='disabled'))
        win.mainloop()
        

    except IOError:
        pass


#button

#labelframe = LabelFrame(win, text="Leaf Disease Detection using OPENCV")
#labelframe.pack(fill="both", expand="yes")
label1 = Label(win, text="GUI For Leaf Disease Detection using OPENCV", fg ='blue')
label1.pack()
    
b1=tk.Button(win, text= 'browse image',width=25, height=3,fg ='red', command=b1_click)
b1.pack()
lbl = Label(win, text="Result", fg ='blue')
lbl.pack()

#image =ImageTk.PhotoImage(file='a.JPG')

#img1='1.JPG'
#lb2 = Label(win,image=image)
#lb2.pack()
#if ("Result" ==('Grape___Black_rot')):
#lse:
   # print ('normal leaf')


#lbl.grid(column=0, row=0)
win.geometry("550x250")
win.title("Leaf Disease Detection using OPENCV")
win.bind("<Return>", b1_click)
win.mainloop() 
