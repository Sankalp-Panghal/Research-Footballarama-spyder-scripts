# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:14:54 2021

@author: panghals
"""

import os
import subprocess


yolo_loc = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64"
print(yolo_loc)
os.chdir(yolo_loc)
command = "darknet detector map data/obj.data cfg/yolov4-customobj.cfg backup1080/yolov4-obj_best.weights".split(" ")

with open("cfg/yolov4-obj.cfg") as f:
    cfg_readlines = f.readlines()

with open("all_size_test_result_1080.txt",'w') as main_f:

    for size in range(672,1088+32,32):
        print("Size - {}".format(size))
        with open("cfg/yolov4-customobj.cfg",'w') as f:
            cfg_readlines[7] = "width = {}\n".format(size)
            cfg_readlines[8] = "height = {}\n".format(size)
            f.writelines(cfg_readlines)

        #--- Main command
        p1 = subprocess.run(command, capture_output=True ,text=True )
        #----------------

        #print("Return code - {}".format(p1.returncode))
        result = "\n".join(p1.stdout.split('\n')[24:31])+"\n"
        print("Result - ",result)
        main_f.write("\n\n\n------------------------{}x{}------------------------\n".format(size,size))
        main_f.write(result)