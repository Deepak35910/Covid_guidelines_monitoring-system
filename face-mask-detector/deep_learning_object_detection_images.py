# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
from detect_mask_image import fun1
from color import color_cloth

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#ap.add_argument("-p", "--prototxt", required=True,
#	help="path to Caffe 'deploy' prototxt file")
#ap.add_argument("-m", "--model", required=True,
#	help="path to Caffe pre-trained model")
#ap.add_argument("-c", "--confidence", type=float, default=0.2,
#	help="minimum probability to filter weak detections")
#args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(r'C:/Users/Deepak/cv2/pyimage/face-mask-detector/MobileNetSSD_deploy.prototxt.txt', r'C:/Users/Deepak/cv2/pyimage/face-mask-detector/MobileNetSSD_deploy.caffemodel')

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
image = cv2.imread('images/without-mask-UNI-3.jpg')
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	if i<=0 or i>=2:
		continue
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > 0.6 and CLASSES[int(detections[0, 0, i, 1])]=='person':
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		img_pass=image[startY:startY+endY,startX:startX+endX]
		label=fun1(img_pass)
		color_label=color_cloth(img_pass)
		color1=()
		if label[0:4] == "Mask":
			color1 = (0, 255, 0)
		else: 
			color1=(0, 0,255 )        
            
		    # Drawing bounding box on the original image
		cv2.rectangle(image, (startX, startY),
		              (endX, endY),
		              color1, 2)
		# Preparing text with label and confidence for current bounding box
		#text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],confidences[i])
		# Putting text with label and confidence on the original image
		#cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
		#            cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)
        
		cv2.putText(image, label, (startX, startY - 5),
		cv2.FONT_HERSHEY_COMPLEX, 0.7, color1, 2)
        
		cv2.putText(image, color_label, (startX+10, startY +10),
		cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,0), 2)
        
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)