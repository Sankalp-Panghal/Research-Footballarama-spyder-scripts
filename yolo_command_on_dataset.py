# -*- coding: utf-8 -*-
"""
Created on Sun May  2 11:49:40 2021

@author: panghals
"""



import os
import subprocess


yolo_loc = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64"
print(yolo_loc)
os.chdir(yolo_loc)

main_loc = "D:\\Sequence Data Feb\\Seq 5 shelly 020"

for no in range(1,2):
    loc = main_loc + "\\" + str(no)
    command = "darknet.exe detector demo data/obj.data cfg/yolov4-detect.cfg backup672/yolov4-obj_best.weights -thresh 0.15 \"{0}\\1080.mp4\" -out_filename \"{0}\\1080_yolo_result.mp4\"".format(loc)#.split(" ")
    print(command)
    subprocess.run(command, shell=True)
    print("Done\n".format(no))