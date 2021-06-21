# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:00:32 2021

Run FUll YOLO with 80 clasees on my custom dataset to see how does it look.

@author: panghals
"""

# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
#import argparse
import time
import cv2
import os
import imutils

def resize(im, h):
    return imutils.resize(im,height = h)

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#     help="path to input image")
# ap.add_argument("-y", "--yolo", required=True,
#     help="base path to YOLO directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#     help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3,
#     help="threshold when applyong non-maxima suppression")
# args = vars(ap.parse_args())

yolo_weights_path = "D:\\SankalpStuff\\YOLO\\yolo-object-detection-pysearch\\yolo-coco" # args["yolo"]
desired_confidence = 0.5 #args["confidence"]
desired_threshold = None #args["threshold"]
frames_locations = ["D:\\Sequence Data Feb\\Seq 5 shelly 020\\1\\720", "D:\\Sequence Data Feb\\Seq 5 shelly 020\\2\\1080" ]

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([yolo_weights_path, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\backup\\train2_best.weights"#os.path.sep.join([yolo_weights_path, "yolov3.weights"])
configPath = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\cmd\\detect2.cfg" # os.path.sep.join([yolo_weights_path, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from {}".format(yolo_weights_path))

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
for frames_location in frames_locations[1:]:

    print("\n"+frames_location.split('\\')[-1])

    image_files = os.listdir(frames_location)
    #image_files = image_files[:1]

    for file in image_files:
        file = os.path.join(frames_location, file)
        image = cv2.imread(file)
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        #------------------------------
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        #------------------------------
        # show timing information on YOLO
        print("YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > desired_confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, desired_confidence, desired_threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
                #text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # show the output image
        cv2.imshow("Image", resize(image,720))
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
        if key == ord('s'):
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key== ord('c'):
                    break

cv2.destroyAllWindows()