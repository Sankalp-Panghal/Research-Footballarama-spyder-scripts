# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:12:42 2021

In YOLO, to test statistics/metrics by command line, we are required to update cfg file (configuration file) with the width and height (resizing done at first step, before testing the image)

In this script
1.For every new size, I update the configuration file (cfg_my_custom_cfg.cfg)
2.Run cmd command for YOLO command to test validation set with this configuration
3.Capture relevant output of command
4.Write to a single result file -> "metric_output.txt", for analysis

@author: panghals
"""

import os
import subprocess as sp

def create_cfg():
    global size, cfg_full_path
    with open(cfg_full_path,'r') as cfg_r:
        with open("cfg/cfg_my_custom_cfg.cfg",'w') as cfg_w:
            line = cfg_r.readline()
            while line:
                if "width" not in line and "height" not in line:
                    cfg_w.write(line)
                else:
                    if "width" in line:
                        t = "width"
                    elif "height" in line:
                        t = "height"
                    else:
                        t = "None"
                    cfg_w.write("{} = {}\n".format(t,size))
                line = cfg_r.readline()

#Program Start
print("Starting path ->",os.getcwd())
yolo_path = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64"
os.chdir(yolo_path)
print("Current path ->",os.getcwd())

cfg_file = "cfg/yolov4-obj.cfg"
weights_path = "backup1080/yolov4-obj_best.weights"
yolo_test_command = "darknet.exe detector map data/obj.data {} {}".format(cfg_file,weights_path)

cfg_full_path = os.path.join(yolo_path,cfg_file)
result_write_path = "metric_output.txt"
size = 416

while size<=416:
    create_cfg()
    cfg_file = "cfg/cfg_my_custom_cfg.cfg"
    yolo_test_command = "darknet.exe detector map data/obj.data {} {} > {}".format(cfg_file,weights_path,result_write_path)
    print("Runnin -> ",yolo_test_command)
    code = sp.run(yolo_test_command.split(" "))
    print("code ",code)
    size += 32





