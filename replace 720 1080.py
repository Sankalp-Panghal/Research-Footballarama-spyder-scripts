# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:46:01 2021

@author: panghals
"""

import os
import shutil

s = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\images"
dest_train = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\train\\"
dest_test = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\test\\"
dest_val = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\val\\"

ff = [dest_test,dest_val]

for f in ff:
    files = [x for x in os.listdir(f) if 'png' in x]
    for file in files:
        shutil.copy(os.path.join(s,file),os.path.join(f,file))

        

