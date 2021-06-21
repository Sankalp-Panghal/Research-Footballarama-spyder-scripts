# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:58:17 2021

A stupid txt file (train.txt) is required to contain names of images in a folder, so I simply run this script to write names of those files in this txt file

@author: panghals
"""
import os

def train(location_traintxt):
    for name in ['train','val']:
        txt_name = location_traintxt + "\\{}_1080.txt".format(name)
        image_folder = location_traintxt + "\\obj\\previous dataset\\{}".format(name)
        txt_prepend = "data/obj/{}/".format(name)
        with open(txt_name,'w') as f:
            images = os.listdir(image_folder)
            for file in images:
                if 'txt' not in file:
                    f.write(txt_prepend+file+'\n')
        print("{}.txt created".format(name))

if __name__ == "__main__":
    location_traintxt = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data"
    #location_traintxt = "D:\\SankalpStuff\\YOLO\\darknet\\build\\darknet\\x64\\data\\obj\\previous dataset\\train"
    train(location_traintxt)