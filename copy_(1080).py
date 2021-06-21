# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 09:40:09 2021

@author: panghals
"""

#---- FFMPEG all sequences -----

import os
import shutil

main_to = "D:\\Sequence Data Feb\\Seq 5 shelly 020"
main_from = "\\\\boyle\\Theta\\Footballarama\\Sankalp Stuff\\DATASET MOTION\\Updated\\dataset"


for no in range(6,11):
    fromm = main_from + "\\{}\\1080".format(no)
    to = main_to + "\\{}\\1080".format(no)

    print(to)

    for file in os.listdir(fromm):
        print(file)
        shutil.copy(os.path.join(fromm,file), os.path.join(to,file))