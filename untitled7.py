# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 09:09:21 2021

@author: panghals
"""

import os
import shutil

name = "train"
loc = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\"
train_txt = loc + "{}.txt".format(name)

s = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\images"
dest_train = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\train\\"
dest_test = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\test\\"
dest_val = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\val\\"

def empty_folder(folder):
    files = os.listdir(folder)
    for file in files:
        os.remove(os.path.join(folder,file))
        
empty_folder(dest_train)

with open(train_txt) as f:
    file = f.readline()
    while file:
        file = file.split('/')[-1].strip()
        shutil.copy(os.path.join(s,file),os.path.join(dest_train,file))
        file = f.readline()