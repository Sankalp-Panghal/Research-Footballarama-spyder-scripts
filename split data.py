# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:17:20 2021

TRAIN - TEST - VAL split

@author: panghals
"""


import random
import os
import shutil


s = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\images split"
dest_train = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\train\\"
#dest_test = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\test\\"
dest_val = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\val\\"


files = [x for x in os.listdir(s) if 'png' in x]

for x in range(3):
    random.shuffle(files)

for i in range(round(.9*len(files))):
    file = files[i]
    frame_no = file.split('.')[0].split('_')[1]
    shutil.copy(s+"\\"+file,dest_train)
    shutil.copy(s+"\\"+"frame_{}.txt".format(frame_no),dest_train)

for i in range(i,len(files)):
    file = files[i]
    frame_no = file.split('.')[0].split('_')[1]
    shutil.copy(s+"\\"+file,dest_val)
    shutil.copy(s+"\\"+"frame_{}.txt".format(frame_no),dest_val)


# for i in range(i,len(files)):
#     file = files[i]
#     frame_no = file.split('.')[0].split('_')[1]
#     shutil.copy(s+"\\"+file,dest_test)
#     shutil.copy(s+"\\"+"frame_{}.txt".format(frame_no),dest_test)

print("split complete")



