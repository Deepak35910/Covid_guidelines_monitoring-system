# USAGE
# python detect_mask_image.py --image examples/example_01.png

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image

def fun1(img_pass):
    ret_startX=0
    ret_endX=0
    ret_startY=0
    ret_endY=0
    label=str()
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image",
        help="path to input image")
    ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
        default="mask_detector_dummy2.model",
        help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.8,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    args["image"]=img_pass
    # load our serialized face detector model from disk
    #print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    

    # load the face mask detector model from disk
    #print("[INFO] loading face mask detector model...")
    model = load_model(args["model"])
    
    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    image = img_pass
    orig = image.copy()
    (h, w) = image.shape[:2]

    #window_name = 'image'
    #cv2.imshow(window_name,img_pass)
    #cv2.waitKey(0)
    
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    #print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()
    #print('mask',1)
    
    # loop over the detections
    for couuunt,i in enumerate(range(0, detections.shape[2])):
        if (couuunt>0):
            continue
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        #print('face prob',confidence,detections[0, 0, i, 3:7]* np.array([w, h, w, h]))
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #print('mask',2)
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            #print('face prob',confidence,np.array([startX,startY,endX,endY])* np.array([w, h, w, h]))
            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            ret_startX,ret_endX,ret_startY,ret_endY =startX,endX,startY,endY
            #print('mask',3)
            face = image[startY:endY, startX:endX]
            #face = cv2.imread(r'C:\Users\Deepak\Desktop\resume\New folder\agreen_mas1k.jpg')
            #face = load_img(r'C:\Users\Deepak\Desktop\resume\New folder\agreen_mas1k.jpg')
            #face = cv2.imread(r'C:\Users\Deepak\Desktop\resume\New folder\agreen_mas1k.jpg')
            
            #window_name = 'image'
            #cv2.imshow(window_name,face)
            #cv2.waitKey(0)
            #     323    159  362    312
            if (endX-startX) <=0:
                label='FB'
                print(label)
                color = (0, 255, 0)
                break
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            #converting into pil
            #face = Image.fromarray(face)
            face = cv2.resize(face,(224, 224))
            #window_name = 'image'
            #cv2.imshow(window_name,face)
            #cv2.waitKey(0)
            face = img_to_array(face)
           
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            #print(face)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]
            #print('mask',4,mask, withoutMask)
            # determine the class label and color we'll use to draw
            # the bounding box and text
            if withoutMask > mask and withoutMask > 0.9:
                label="No Mask"
                #window_name = 'image'
                #cv2.imshow(window_name,img_pass)
                #cv2.waitKey(0)
                #print(label)
                color =(0, 0, 255)
            if mask > withoutMask :
                label="Mask"
                #print(label)
                color = (0, 255, 0)

                
            # include the probability in the label
            #label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            #window_name = 'image'
            #cv2.rectangle(img_pass, (ret_startX, ret_startY),
            #         (ret_endX, ret_endY),
            #         (0,0,255), 2)
            #cv2.imshow(window_name,img_pass)
            #cv2.waitKey(0)      
        else:
            label='FND'
            print(label)
            color = (0, 255, 0)
        print(label)
    return label,ret_startX,ret_endX,ret_startY,ret_endY
#		# display the label and bounding box rectangle on the output
#		# frame
#		cv2.putText(image, label, (startX, startY - 10),
#			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
#		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# show the output image
#cv2.imshow("Output", image)
#cv2.waitKey(0)