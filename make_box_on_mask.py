# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 08:21:51 2021

GENERATE BOUNDING BOXES FOR MASKS

@author: panghals
"""
import cv2
#import imutils
import numpy as np
import os
from scipy.spatial import distance



#def resize(im,h):
#    return imutils.resize(im,height = h)


mask_loc = "D://Sequence Data Feb//Seq 5 shelly 020//1//masks_720//"
frame_loc = "D:\\Sequence Data Feb\\Seq 5 shelly 020\\1\\frames_720\\"

start_no = 8170
end_no = start_no + 501

# rows,cols,dim = mask.shape
# print(mask.shape)
# cv2.imshow("mask",mask)
# cv2.waitKey(0)

for no in range(start_no, end_no):

    frame = cv2.imread(frame_loc+"frame_{}.png".format(no))
    mask = cv2.cvtColor(cv2.imread(mask_loc+"mask_{}.png".format(no)) , cv2.COLOR_BGR2GRAY)
    
    #ret,thresh = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(mask, 1, 2)
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    cv2.imshow("CV2 auto box generate using mask",frame)
    cv2.waitKey(40)
    
cv2.destroyAllWindows()
print("Script finish")




