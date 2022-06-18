######## Car Detector Camera using TensorFlow Object Detection API #########
# Part of code copied from: Evan Juras
# Date: 10/15/18
#
# The framework is based off the Object_detection_picamera.py script located here:
# https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py
#
# This code was written by eneroboy
# github.com/eneroboy

### TensorFlow version 2.2.0
### OpenCV version 4.2.0
### NumPy version 1.17.4

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import sys

# Set up camera constants
IM_WIDTH = 480
IM_HEIGHT = 640

#Select camera type (if user enters --usbcam when calling this script,
#a USB webcam will be used)
camera_type = 'usb'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam', action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

#### Initialize TensorFlow model ####

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

## Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#### Initialize other parameters ####

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Define inside box coordinates (top left and bottom right)
TL_inside = (int(IM_WIDTH*0),int(IM_HEIGHT*0.25))
BR_inside = (int(IM_WIDTH*0.31),int(IM_HEIGHT*0.85))

# Define outside box coordinates (top left and bottom right)
TL_outside = (int(IM_WIDTH*0.31),int(IM_HEIGHT*0.25))
BR_outside = (int(IM_WIDTH*0.8),int(IM_HEIGHT*.85))

# Initialize control variables used for car detector
detected_inside = False
detected_outside = False

inside_counter = 0
outside_counter = 0

pause = 0
pause_counter = 0

#### Car detection function ####

# This function contains the code to detect a car, determine if it
# came or left, and will send a text to the user's phone.
def car_detector(frame):

    # Use globals for the control variables so they retain their value after function exits
    global detected_inside, detected_outside
    global inside_counter, outside_counter
    global pause, pause_counter

    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = \
        sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        camera.get(1),
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.40)

    # Draw boxes defining "outside" and "inside" locations.
    cv2.rectangle(frame,TL_outside,BR_outside,(255,20,20),3)
    cv2.putText(frame,"Right box",(TL_outside[0]+10,TL_outside[1]-10),font,1,(255,20,255),3,cv2.LINE_AA)
    cv2.rectangle(frame,TL_inside,BR_inside,(20,20,255),3)
    cv2.putText(frame,"Left box",(TL_inside[0]+10,TL_inside[1]-10),font,1,(20,20,20),3,cv2.LINE_AA)
    
    # Check the class of the top detected object by looking at classes[0][0].
    # find its center coordinates by looking at the boxes[0][0] variable.
    # boxes[0][0] variable holds coordinates of detected objects as (ymin, xmin, ymax, xmax)
    if (((int(classes[0][0]) == 3) or (int(classes[0][0] == 1) )) and (pause == 0)):
        x = int(((boxes[0][0][1]+boxes[0][0][3])/2)*IM_WIDTH)
        y = int(((boxes[0][0][0]+boxes[0][0][2])/2)*IM_HEIGHT)

        # Draw a circle at center of object
        cv2.circle(frame,(x,y), 5, (75,13,180), -1)
        # If object is in inside box, increment inside counter variable
        if ((x > TL_inside[0]) and (x < BR_inside[0]) and (y > TL_inside[1]) and (y < BR_inside[1])):
            inside_counter = inside_counter + 1

        # If object is in outside box, increment outside counter variable
        if ((x > TL_outside[0]) and (x < BR_outside[0]) and (y > TL_outside[1]) and (y < BR_outside[1])):
            outside_counter = outside_counter + 1

    # If car has been detected inside for more than 10 frames, set detected_inside flag
    first = 0
    if inside_counter > 10:
        first = 1
        if outside_counter < 10:
            detected_inside = True
            inside_counter = 0
            outside_counter = 0
            # Pause car detection by setting "pause" flag
            pause = 1

    # If car has been detected outside for more than 10 frames, set detected_outside flag
    if outside_counter > 10:
        if inside_counter > 10:
            detected_outside = True
            inside_counter = 0
            outside_counter = 0
            # Pause car detection by setting "pause" flag
            pause = 1

    # If pause flag is set, draw message on screen.
    if pause == 1:
        if detected_inside == True:
            cv2.putText(frame,'Car left',(int(IM_WIDTH*.1),int(IM_HEIGHT*.95)),font,1,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(frame,'Car left!',(int(IM_WIDTH*.1),int(IM_HEIGHT*.95)),font,1,(95,176,23),3,cv2.LINE_AA)

        if detected_outside == True:
            cv2.putText(frame,'Car came',(int(IM_WIDTH*.1),int(IM_HEIGHT*.95)),font,1,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(frame,'Car came',(int(IM_WIDTH*.1),int(IM_HEIGHT*.95)),font,1,(95,176,23),3,cv2.LINE_AA)

        # Increment pause counter until it reaches 60 (for a framerate of 8.0 FPS, this is about 7.5 seconds),
        # then unpause the application (set pause flag to 0).
        pause_counter = pause_counter + 1
        if pause_counter > 60:
            pause = 0
            pause_counter = 0
            detected_inside = False
            detected_outside = False

    # Draw counter info
    cv2.putText(frame,'Detection counter: ' + str(max(inside_counter,outside_counter)),(250,70),font,0.5,(128,128,0),1,cv2.LINE_AA)
    cv2.putText(frame,'Pause counter: ' + str(pause_counter),(250,50),font,0.5,(128,128,0),1,cv2.LINE_AA)

    return frame

#### Initialize camera and perform object detection ####

### USB webcam ###
    
if camera_type == 'usb':
    # Initialize USB webcam feed
    #camera = cv2.VideoCapture('http://192.168.0.134:4747/video')
    camera = cv2.VideoCapture('zafira1.mp4')
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)

    # Continuously capture frames and perform object detection on them
    while(True):

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        (ret, frame) = camera.read()

        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # Pass frame into pet detection function
        frame = car_detector(frame)

        # Draw FPS
        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(128,128,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # FPS calculation
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
        
cv2.destroyAllWindows()