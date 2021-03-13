from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pandas as pd
from keras.models import model_from_json

def color_cloth(img_pass,x1,x2,y1,y2):
    #img = cv2.imread('black_tshirt.jpg')
    print('inside color shape of image is  -  ',img_pass.shape,x1,x2,y1,y2)
    img = img_pass
    height, width, dim = img.shape
    print(height, width)
    #img = img[y1:y2, x1:x2, :]
    #img = img[:, 0:int((2.5*width/4)), :]
    #img = img[int((height/4)):int((2*height/4)), int((width/4)):int((3*width/4)), :]
    #img = img[int((height/4)):int((2*height/4)), int((width/4)):int((3*width/4)), :]
    #img = img[int((0.5*(height/4))):int((3*(height/4))), int((1*(width/4))):int((3*(width/4))), :]

    #img=cv2.imread('C:/Users/Deepak/cv2/pyimage/face-mask-detector/images/white00.jpg')
    #print(img[0][0])
    #window_name = 'image'
    #cv2.imshow(window_name,img)
    #cv2.waitKey(0)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print(img[0])
    img=cv2.resize(img, (60, 160),interpolation=0) 
    print(img[0][0])
    img=img/255
    print(img[0][0])
    height, width, dim = img.shape
    #print(height, width)
    #window_name = 'image'
    #cv2.imshow(window_name,img)
    #cv2.waitKey(0)  
    
    # load json and create model
    json_file = open('color_weights1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("color_weights1.h5")
    print("Loaded color model from disk")

    pred=loaded_model.predict_generator(np.expand_dims(img,axis=0),verbose=1)
    print(pred)
    predicted_class_indices=np.argmax(pred,axis=1)
    number=int(predicted_class_indices)
    dict1={0: 'Black',
     1: 'Blue',
     2: 'Gray',
     3: 'Green',
     4: 'Magenta',
     5: 'Maroon',
     6: 'Red',
     7: 'White',
     8: 'Yellow',
     9: 'brown',
     10: 'khaki',
     11: 'orange',
     12: 'pink'}
    print(dict1[number],np.max(pred))
    return dict1[number]

#img5 = cv2.imread('chk.jpg')
#color_label=color_cloth(img5)
#print(color_label)
