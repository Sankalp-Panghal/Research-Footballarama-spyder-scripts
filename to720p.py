# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:16:32 2021

@author: panghals
"""

import os
import imutils 
import cv2
import shutil

def resize(im,h):
    return imutils.resize(im,height = h)

location = "D:\\Sequence Data Feb\\Seq 5 shelly 020\\1\\"
img_loc = location + "frames\\" 
yolo_loc =  location + "yolo format ball\\" 
files = os.listdir(yolo_loc)
write = location+"1080\\"

for file in files:
    if file=="classes.txt":
        continue
    print(file)
    frame_no = file.split('.')[0].split('_')[1]
    cv2.imwrite(write + "frame_{}.png".format(frame_no), resize(cv2.imread(img_loc+ "frame_{}.tiff".format(frame_no)),1080 ))
print("done")
        