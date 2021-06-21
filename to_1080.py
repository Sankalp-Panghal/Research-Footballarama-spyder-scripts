# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:44:28 2021

@author: panghals
"""
import os
import shutil
import imutils
import cv2


def resize(im, h):
    return imutils.resize(im,height = h)

loc = "\\\\boyle\\Theta\\Footballarama\\Sankalp Stuff\\DATASET MOTION\\Updated\\dataset\\"

for no in range(9,11):
    loc_no = loc + str(no)
    os.chdir(loc_no)
    try:
        os.mkdir("1080")
    except:
        pass
    frames_loc = loc_no + "\\frames"
    frames = os.listdir(frames_loc)
    for fr in frames:
        im = cv2.imread(os.path.join(frames_loc,fr))
        im = resize(im,1080)
        cv2.imwrite( loc_no + "\\1080\\" +fr.split('.')[0] +'.png' ,im )
        print("{}--->{}".format(no,fr))


