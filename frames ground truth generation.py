# -*- coding: utf-8 -*-
"""
Created on Sun May  2 09:56:54 2021

@author: panghals
"""

import cv2
import subprocess
import os

def find_start_no(loc):
    files = sorted([int(x.split('.')[0].split('_')[1]) for x in os.listdir(loc) if 'png' in x])
    return (files[0], len(files))

def extract_xywh(location_of_yolo_truth_file):

    try:
        with open(location_of_yolo_truth_file) as f:
            line = f.readline()
            if line:
                class_obj, x_cen, y_cen, w, h = [float(x) for x in line.split()]
                w = int(w*cols); h = int(h*rows); x = int(x_cen*cols - (w/2)) ;y = int(y_cen*rows - (h/2)) ;
            else:
                x,y,w,h = None, None, None, None
            #print("frame {} yolo read ----> class {},  x_cen {},  y_cen {},  w {},  h {}".format(frame_no, class_obj, x_cen, y_cen, w, h))

    except:
        print("Cannot open file --> {}".format(location_of_yolo_truth_file))
        x,y,w,h = None, None, None, None

    return (x,y,w,h)


main_loc = "D:\\Sequence Data Feb\\Seq 5 shelly 020"

cols = 2048; rows = 1080;

for no in range(1,2):
    fold = main_loc + "\\" + str(no)
    frames_loc = fold + "\\1080"
    ground_truth = fold + "\\yolo format ball"
    write_loc = fold + "\\1080 ground truth"

    os.chdir(fold)
    if "1080 ground truth" not in os.listdir(fold):
        os.mkdir("1080 ground truth")

    for yolo_file in [x for x in os.listdir(ground_truth) if "class" not in x]:
        xywh = extract_xywh(ground_truth+"\\"+yolo_file)
        im = cv2.imread( frames_loc +"\\" + yolo_file.split(".")[0]+".png")
        if xywh[0]:
            x,y,w,h = xywh
            cv2.rectangle(im, (x,y),(x+w,y+h) , (0,255,0), 2)
        print(frames_loc +"\\" + yolo_file.split(".")[0]+".png")
        #cv2.imshow(frames_loc +"\\" + yolo_file.split(".")[0]+".png",im)
        #cv2.waitKey(0)
        cv2.imwrite( write_loc + "\\" + yolo_file.split(".")[0]+".png" , im)


