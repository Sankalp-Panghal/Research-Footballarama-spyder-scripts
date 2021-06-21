# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:59:40 2021

CSV to YOLO
@author: panghals
"""


import cv2 
import imutils

def resize(im,h):
    return imutils.resize(im,height = h)

#-------------------------------------------SPECIFY MANUALLY---------------------------------------------------------
location = "D:\\Sequence Data Feb\\Seq 5 shelly 020\\3\\"
#-------------------------------------------------------------------------------------------------------------------
start_frame_no = 39983

img_loc = location + "frames\\" 

img = cv2.imread(img_loc+"frame_"+str(start_frame_no) +'.tiff')
rows,cols,dim = img.shape

yolo_loc = location + "yolo format ball\\"

class_no = 0

with open(location+"bally.csv",'r') as ball:
    ball.readline()
    line = ball.readline()
    while line:
        frame_no,x,y,w,h = [int(x) for x in line.split(",")]
        x = (x + (w/2))/cols
        y = (y + (h/2))/rows
        w = w/cols
        h = h/rows
        with open(yolo_loc +"frame_"+ str(frame_no) +'.txt', 'w') as file:
            if x>0:
                file.write("{} {} {} {} {}".format(class_no,x,y,w,h))
            else:
                file.write("")
        line = ball.readline()
        
print("Script yolo format done")