# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:59:05 2021

Refine 1

@author: panghals
"""
import os
import cv2 
import imutils

def resize(im,h):
    return imutils.resize(im,height = h)

#-------------------------------------------SPECIFY MANUALLY---------------------------------------------------------
location = "D:\\Sequence Data Feb\\Seq 5 shelly 020\\1\\"
#------------------------------------------------------------------------

yolo_loc = location + "yolo format ball"

files = os.listdir(yolo_loc)
removed = 0

for file in files:
    with open(yolo_loc+"\\"+file) as f:
        content = f.read()
    if len(content)==0:
        removed += 1
        os.remove(yolo_loc+"\\"+file)
        
print("{} files removed".format(removed))