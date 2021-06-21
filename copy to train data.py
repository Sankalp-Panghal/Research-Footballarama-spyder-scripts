# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:26:27 2021

copy to training folder

@author: panghals
"""

import os
import shutil

def empty_folder(folder):
    files = os.listdir(folder)
    for file in files:
        os.remove(os.path.join(folder,file))

dest_train = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\images"

empty_folder(dest_train)

source_loc  = "D:\\Sequence Data Feb\\Seq 5 shelly 020\\"
source = ["1","extra framez","extra frames 2"]

for s in source:
    yolo_loc = source_loc + s + "\\yolo format ball"
    loc_img = source_loc + s + "\\1080\\"
    
    files = os.listdir(yolo_loc)
    
    for file in files:
        if file=="classes.txt":
            continue
        frame_no = file.split('.')[0].split('_')[1]
        
        shutil.copy(yolo_loc+"\\"+file,dest_train)
        shutil.copy(loc_img + "frame_{}.png".format(frame_no), dest_train)
        print("Copying ----> {}".format(yolo_loc+"\\"+file))
        
        