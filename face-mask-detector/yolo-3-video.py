

"""
Course:  Training YOLO v3 for Objects Detection with Custom Data

Section-2
Objects Detection on Video with YOLO v3 and OpenCV
File: yolo-3-video.py
"""


# Detecting Objects on Video with OpenCV deep learning library
#
# Algorithm:
# Reading input video --> Loading YOLO v3 Network -->
# --> Reading frames in the loop --> Getting blob from the frame -->
# --> Implementing Forward Pass --> Getting Bounding Boxes -->
# --> Non-maximum Suppression --> Drawing Bounding Boxes with Labels -->
# --> Writing processed frames
#
# Result:
# New video file with Detected Objects, Bounding Boxes and Labels


# Importing needed libraries
import numpy as np
import cv2
import time
import os
from detect_mask_image import fun1
from color import color_cloth
#from face-mask-detector import detect_mask_image

from gtts import gTTS
import os
language='en'
from datetime import datetime

"""
Start of:
Reading input video
"""

# Defining 'VideoCapture' object
# and reading video from a file
# Pay attention! If you're using Windows, the path might looks like:
# r'videos\traffic-cars.mp4'
# or:
# 'videos\\traffic-cars.mp4'
video = cv2.VideoCapture('images/mask_blue.avi')
# Preparing variable for writer
# that we will use to write processed frames
writer = None

# Preparing variables for spatial dimensions of the frames
h, w = None, None

"""
End of:
Reading input video
"""


"""
Start of:
Loading YOLO v3 network
"""

# Loading COCO class labels from file
# Opening file
# Pay attention! If you're using Windows, yours path might looks like:
# r'yolo-coco-data\coco.names'
# or:
# 'yolo-coco-data\\coco.names'
with open('yolo-coco-data/coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]


# # Check point
# print('List with labels names:')
# print(labels)

# Loading trained YOLO v3 Objects Detector
# with the help of 'dnn' library from OpenCV
# Pay attention! If you're using Windows, yours paths might look like:
# r'yolo-coco-data\yolov3.cfg'
# r'yolo-coco-data\yolov3.weights'
# or:
# 'yolo-coco-data\\yolov3.cfg'
# 'yolo-coco-data\\yolov3.weights'
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')

# Getting list with names of all layers from YOLO v3 network
layers_names_all = network.getLayerNames()

# # Check point
# print()
# print(layers_names_all)

# Getting only output layers' names that we need from YOLO v3 algorithm
# with function that returns indexes of layers with unconnected outputs
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# # Check point
# print()
# print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.9

# Setting threshold for filtering weak bounding boxes
# with non-maximum suppression
threshold = 0.2

# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# # Check point
# print()
# print(type(colours))  # <class 'numpy.ndarray'>
# print(colours.shape)  # (80, 3)
# print(colours[0])  # [172  10 127]

"""
End of:
Loading YOLO v3 network
"""


"""
Start of:
Reading frames in the loop
"""
Overall_skip_frame=0
skip_frame=0
skip_color_label=''
# Defining variable for counting frames
# At the end we will show total amount of processed frames
f = 0
counter=0
# Defining variable for counting total time
# At the end we will show time spent for processing all frames
t = 0
# Defining loop for catching frames
while True:
    # Capturing frame-by-frame
    ret, frame = video.read()
    counter=counter+1
    Overall_skip_frame=Overall_skip_frame+1
    # If the frame was not retrieved
    # e.g.: at the end of the video,
    # then we break the loop
    if not ret:
        break
        
    if (skip_frame>=1 and skip_frame<=60 ):
        cv2.putText(frame, str('People wearing a '+skip_color_label+' color cloth'), (0,int(h*0.45)), 0, 1.5,(0, 255, 255), 2)
        cv2.putText(frame, str('Please wear a mask'), (0,int(h*0.55)), 0, 1.5,(0, 255, 255), 2)
        
        skip_frame=skip_frame+1
        print('skip_frame  at top ',skip_frame,color_label)

        writer.write(frame)
       
        continue
    skip_frame=0    
    skip_color_label=''

    if (Overall_skip_frame % 60 !=0):
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter('images/result.mp4', fourcc, 20,
                                 (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)
        continue
    # Getting spatial dimensions of the frame
    # we do it only once from the very beginning
    # all other frames have the same dimension
    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = frame.shape[:2]

    """
    Start of:
    Getting blob from current frame
    """

    # Getting blob from current frame
    # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob from current
    # frame after mean subtraction, normalizing, and RB channels swapping
    # Resulted shape has number of frames, number of channels, width and height
    # E.G.:
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    """
    End of:
    Getting blob from current frame
    """

    """
    Start of:
    Implementing Forward pass
    """

    # Implementing forward pass with our blob and only through output layers
    # Calculating at the same time, needed time for forward pass
    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Increasing counters for frames and total time
    f += 1
    t += end - start
    # Showing spent time for single current frame
    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))
    """
    End of:
    Implementing Forward pass
    """

    """
    Start of:
    Getting bounding boxes
    """

    # Preparing lists for detected bounding boxes,
    # obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # # Check point
            # # Every 'detected_objects' numpy array has first 4 numbers with
            # # bounding box coordinates and rest 80 with probabilities
            #  # for every class
            # print(detected_objects.shape)  # (85,)

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum and class_current==0:
                # Scaling bounding box coordinates to the initial frame size
                # YOLO data format keeps coordinates for center of bounding box
                # and its current width and height
                # That is why we can just multiply them elementwise
                # to the width and height
                # of the original frame and in this way get coordinates for center
                # of bounding box, its width and height for original frame
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    """
    End of:
    Getting bounding boxes
    """

    """
    Start of:
    Non-maximum suppression
    """

    # Implementing non-maximum suppression of given bounding boxes
    # With this technique we exclude some of bounding boxes if their
    # corresponding confidences are low or there is another
    # bounding box for this region with higher confidence

    # It is needed to make sure that data type of the boxes is 'int'
    # and data type of the confidences is 'float'
    # https://github.com/opencv/opencv/issues/12789
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

    """
    End of:
    Non-maximum suppression
    """

    """
    Start of:
    Drawing bounding boxes and labels
    """

    # Checking if there is at least one detected object
    # after non-maximum suppression
    if len(results) > 0:
        a=np.array(results.flatten())
        dist=np.array(bounding_boxes)[a]
        mid_bottom=[]
        for i in dist:
            mid_bottom.append((int(i[0]+(0.5*i[2])),(i[1]+i[3])))
        mid_bottom
        # Going through indexes of results
        for y,i in enumerate(results.flatten()):
            # Getting current bounding box coordinates,
            # its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            x_min=max(x_min,0)
            y_min=max(y_min,0)
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            #print(confidences[i])
            img_pass=frame[y_min:y_min+box_height,x_min:x_min+box_width]
            #print('yolo',y_min,y_min+box_height,x_min,x_min+box_width)

            label=''
            label,ret_startX,ret_endX,ret_startY,ret_endY=fun1(img_pass)
            
            #if (ret_endY ==0):
            #    continue
            if (label=='Mask' or label=='FND' or label=='FB' or (ret_startX ==0 and ret_endX  ==0 and ret_startY  ==0 and ret_endY ==0 )):
                print(1)
                continue
            print(2)   
            
            #print(label,ret_startX,ret_endX,ret_startY,ret_endY)
            #window_name = 'image'
            #cv2.imshow(window_name,img_pass)
            #cv2.waitKey(0)            
            #color_label=color_cloth(img_pass,ret_startX,ret_endX,ret_endY,ret_startY+ret_endY+ret_endY)
            print('label is',label)
            color_label=color_cloth(img_pass,ret_startX,ret_endX,ret_startY,ret_startY+ret_endY+ret_endY+ret_endY)
            
            color1=()
            if label[0:4] == "Mask":
                color1 = (0, 255, 0)
            else: 
                color1=(0, 0,255 )
               
            # Preparing colour for current bounding box
            # and converting from numpy array to list
            colour_box_current = colours[class_numbers[i]].tolist()
            
            # # # Check point
            # print(type(colour_box_current))  # <class 'list'>
            # print(colour_box_current)  # [172 , 10, 12
        
            # Drawing bounding box on the original current frame
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          color1, 2)
            
            # Putting text with label and confidence on the original image
            #cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
            #            cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

            cv2.putText(frame, str('People wearing a '+color_label+' color cloth'), (0,int(h*0.45)), 0, 1.5,(0, 255, 255), 1)
            cv2.putText(frame, str('Please wear a mask'), (0,int(h*0.55)), 0, 1.5,(0, 255, 255), 1)
            date_string = datetime.now().strftime("%d%m%Y%H%M%S")
            filename = "Mask_"+color_label+date_string+".mp3"
            mytext='People wearing a '+color_label+' color cloth Please wear a mask'
            output=gTTS(text=mytext,lang=language,slow=False)
            output.save(filename)
            os.system(filename)
            
            #cv2.putText(frame, label, (x_min, y_min - 5),
            #            cv2.FONT_HERSHEY_COMPLEX, 0.7, color1, 2)

            #cv2.putText(frame, color_label, (x_min+10, y_min +10),
            #            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,0), 2)
            skip_frame=1
            print('skip_frame - ',skip_frame,color_label)
            skip_color_label=color_label
            break

        
        
    """
    End of:
    Drawing bounding boxes and labels
    """

    """
    Start of:
    Writing processed frame into the file
    """

    # Initializing writer
    # we do it only once from the very beginning
    # when we get spatial dimensions of the frames
    if writer is None:
        # Constructing code of the codec
        # to be used in the function VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Writing current processed frame into the video file
        # Pay attention! If you're using Windows, yours path might looks like:
        # r'videos\result-traffic-cars.mp4'
        # or:
        # 'videos\\result-traffic-cars.mp4'
        writer = cv2.VideoWriter('images/result.mp4', fourcc, 20,
                                 (frame.shape[1], frame.shape[0]), True)

    # Write processed current frame to the file
    writer.write(frame)

    """
    End of:
    Writing processed frame into the file
    """

"""
End of:
Reading frames in the loop
"""


# Printing final results
print()
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))


# Releasing video reader and writer
video.release()
writer.release()


"""
Some comments

What is a FOURCC?
    FOURCC is short for "four character code" - an identifier for a video codec,
    compression format, colour or pixel format used in media files.
    http://www.fourcc.org


Parameters for cv2.VideoWriter():
    filename - Name of the output video file.
    fourcc - 4-character code of codec used to compress the frames.
    fps	- Frame rate of the created video.
    frameSize - Size of the video frames.
    isColor	- If it True, the encoder will expect and encode colour frames.
"""
