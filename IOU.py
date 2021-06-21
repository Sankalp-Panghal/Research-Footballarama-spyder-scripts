# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:50:43 2021

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

class stats():
    def __init__(self):
        self.n = 0
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        self.avg_IOU = 0

    def update(self,ground_pred, our_pred):

        if len(ground_pred) ==0 and len(our_pred)==0:
            self.TN += 1
        elif len(ground_pred)==0 and len(our_pred)!=0:
            self.FP += len(our_pred)
        elif len(ground_pred)!=0 and len(our_pred)==0:
            self.FN += len(ground_pred)
        else:
            for op in our_pred:
                iou = calc_iou(ground_pred,op)
                if iou == 0:
                    self.FP += 1
                else:
                    self.TP += 1
                self.avg_IOU = self.avg_IOU*(self.n/(self.n+1)) + (iou/(self.n+1))
                self.n += 1

    def stat_print(self):
        print("Total Predictions - {}".format(self.n))
        print("True Positives - {}".format(self.TP))
        print("False Positives - {}".format(self.FP))
        print("True Negatives - {}".format(self.TN))
        print("False Negatives - {}".format(self.FN))
        print("Average IOU - {}".format(self.avg_IOU))

def resize(im, h):
    return imutils.resize(im,height = h)

def calc_iou(a,b):    #x1 y1 (top left)  x2 y2 (bottom right)
    boxA = (a[0],a[1],a[0]+a[2],a[1]+a[3]); boxB = (b[0], b[1], b[0]+b[2],b[1]+b[3]);
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1]); xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3]);
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1])) ;boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

desired_confidence = 0.5 #args["confidence"]
desired_threshold = None #args["threshold"]
frames_location = "D:\\Sequence Data Feb\\Seq 5 shelly 020\\9\\frames"
ground_truth_location = "D:\\Sequence Data Feb\\Seq 5 shelly 020\\9\\yolo format ball"

#---------------------------------------------------------------------------------------------------
our_weightsPath = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\backup1080\\yolov4-obj_best.weights"
our_configPath = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\cfg\\yolov4-detect.cfg"
old_weightsPath = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\yolov4.weights"
old_configPath = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\cfg\\yolov4-original.cfg"
#----------------------------------------------------------------------------------------
our_classes = ["ball"]
names_location = "D:\\SankalpStuff\\YOLO\\yolo-object-detection-pysearch\\yolo-coco\\coco.names"
with open(names_location) as f:
    old_classes = f.read().strip('\n').split('\n')
#---------------------------------------------------------------------------------------------------


# load our YOLO object detector trained on COCO dataset (80 classes)

old_net = cv2.dnn.readNetFromDarknet(old_configPath, old_weightsPath)
our_net = cv2.dnn.readNetFromDarknet(our_configPath, our_weightsPath)
classId_bycolorid = {}

# load our input image and grab its spatial dimensions
print("\n"+frames_location.split('\\')[-1])

files = os.listdir(frames_location)
image_files = [x for x in files if '.txt' not in x]
#image_files = image_files[:1]

old_stat = stats()
our_stat = stats()


for img_name in image_files:

    #------------------Read Image--------------------------------------------------------
    image = cv2.imread(os.path.join(frames_location, img_name))
    old_pred_xywh = []
    our_pred_xywh = []
    #-------------------Open Ground Truth--------------------------------------------
    with open(os.path.join(ground_truth_location,img_name.split('.')[0]+'.txt')) as f_gt:
        line = f_gt.readline()
        if line:
            ground_truth = tuple([float(x) for x in line.split(" ")][1:])
        else:
            ground_truth = []

    for net_number, net in enumerate([old_net,our_net]):
        classes = our_classes if net_number == 1 else old_classes
        model_name = "our" if net_number == 1 else "old"
        pred = our_pred_xywh if net_number == 1 else old_pred_xywh



        (H, W) = image.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        #------------------------------
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        #------------------------------
        print("Frame {} YOLO {} took {:.6f} seconds".format(img_name , model_name ,end - start))

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
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    #print("Found {}".format(classes[classID]))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, desired_confidence, desired_threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                if 'ball' in classes[classIDs[i]]:
                    pred.append((x,y,w,h))
                    #cv2.rectangle(image,( gr_x, gr_y), (gr_x + gr_w, gr_y + gr_h), (0,255,0), 2 )

                    #ground_xywh = (gr_x, gr_y, gr_w, gr_h)
                    #print("IOU -> {}".format(find_IOU(pred,ground_truth)))
                #text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if ground_truth:
        rows,cols = image.shape[:2]
        gr_x, gr_y, gr_w, gr_h = ground_truth
        gr_w=int(gr_w*cols); gr_h=int(gr_h*rows); gr_x = int(gr_x*cols - (gr_w/2)); gr_y = int(gr_y*rows - (gr_h/2));
        #cv2.rectangle(image,(gr_x,gr_y),(gr_x+gr_w,gr_y+gr_h),(0,255,0),2)
    for pred in [old_pred_xywh, our_pred_xywh]:
        color = (0,0,255) if pred==old_pred_xywh else (255,0,0)
        for (x,y,w,h) in pred:
            cv2.rectangle(image,(x,y),(x+w,y+h),color,10)

    old_stat.update(ground_truth,old_pred_xywh)

    our_stat.update(ground_truth,our_pred_xywh)


    cv2.imshow("Image", resize(image,720) )
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
    if key == ord('s'):
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key== ord('c'):
                break

print("OLD Model STATS")
old_stat.stat_print()
print("\n\n")
print("OUR Model STATS")
our_stat.stat_print()
print("\n\n")
cv2.destroyAllWindows()