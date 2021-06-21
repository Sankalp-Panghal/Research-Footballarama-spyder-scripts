# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 08:36:46 2021

YOLO training command

@author: panghals
"""

import os
import subprocess as sp
#from create_train import train

def train(location_traintxt):
    for name in ['train']:
        txt_name = location_traintxt + "\\{}.txt".format(name)
        image_folder = location_traintxt + "\\obj\\{}".format(name)
        txt_prepend = "data/obj/{}/".format(name)
        with open(txt_name,'w') as f:
            images = os.listdir(image_folder)
            for file in images:
                if 'txt' not in file:
                    f.write(txt_prepend+file+'\n')
        print("{}.txt created".format(name))

if __name__ == "__main__":
    darknet_location = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64"
    os.chdir(darknet_location)

    obj_data = "cmd/obj.data"
    train_txt = darknet_location + "\\data"

    width = 948
    height = 500

    # Setup train.txt file
    train(train_txt)
    #print("train.txt created...")

    # Evaluate anchors and use in cfg
    #anchor_command = "darknet.exe detector calc_anchors {} -num_of_clusters {} -width {} -height {}".format(obj_data,9,width,height)
    #sp.run(anchor_command, shell=True)
    #print("Anchors calcualted ...")

    with open(darknet_location+"\\anchors.txt") as f:
        anchors = f.read().strip()
    print(anchors)

    with open(darknet_location+"\\cmd\\train.cfg") as f:
        cfglines = f.readlines()

    with open(darknet_location+"\\cmd\\train2.cfg",'w') as f:
        for line in cfglines:
            if 'width' in line or 'height' in line or 'anchor' in line:
                if 'anchor' in line:
                    l = "anchor = "+anchors+"\n"
                elif 'width' in line:
                    l = "width = {}\n".format(width)
                else:
                    l = "height = {}\n".format(height)
            f.write(line)

    print("Anchors evaluated...")

    train_cfg = "cmd/train2.cfg"
    test_cfg = "cmd/detect.cfg"
    conv137 = "yolov4.conv.137"

    sp.run("darknet.py detector train {} {} {} -map".format(obj_data,train_cfg,conv137 ))
    # with open('test.log', 'wb') as f:
    #     process = sp.Popen("darknet.py detector train {} {} {} -map", shell=True, stdout=sp.PIPE)
    #     for c in iter(lambda: process.stdout.read(1), b''):
    #         sys.stdout.buffer.write(c)
    #         f.buffer.write(c)