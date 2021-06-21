# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 09:18:26 2021

@author: panghals
"""

import os
import shutil

s = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\images"
dest_train = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\train\\"
files = os.listdir(dest_train)

for file in files:
    file = file.split('.')[0] + '.txt'
    shutil.copy(os.path.join(s,file), os.path.join(dest_train,file))