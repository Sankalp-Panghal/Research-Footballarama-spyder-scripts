"""
Created on Wed April 23 10:32:00 2021

A good one

Nice script to make size X size dataset out of existing dataset
I randomly crop section of size x size around the ball (we know gorund truth of ball), to create new dataset for training YOLO


@author: panghals
"""



#----------------------- Extract 672x672 -----------------------------------------------

import os
import cv2
import imutils
import random


def resize(im, h):
    return imutils.resize(im,height = h)


def extract_672_xy(ball_box):
    x,y,w,h = ball_box

    X = x - (new_size/2); Y = y - (new_size/2);
    if x-(new_size/2) < 0 or x+(new_size/2)>cols or y-(new_size/2)<0 or y+(new_size/2)>rows: #Ball at edge
        if x-(new_size/2)<0:
            X = 0
        elif x+(new_size/2)>cols:
            X = cols-new_size
        else:
            X = x-(new_size/2)
        if y-(new_size/2)<0:
            Y = 0
        elif y+(new_size/2)>rows:
            Y = rows-new_size
        else:
            Y = y-(new_size/2)
    else:
        X = X + (random.random()*250)*(1 if random.random() > 0.5 else -1);
        Y = Y + (random.random()*250)*(1 if random.random() > 0.5 else -1);
        while X<0 or X>cols-new_size or Y<0 or Y>rows-new_size:
            X = x + (random.random()*250)*(1 if random.random() > 0.5 else -1);
            Y = y + (random.random()*250)*(1 if random.random() > 0.5 else -1);
    return (int(X),int(Y))

def extract_xywh(location_of_yolo_truth_file):
    try:
        with open(location_of_yolo_truth_file) as f:
            line = f.readline()
            if line:
                class_obj, x_cen, y_cen, w, h = [float(x) for x in line.split()]
                w = int(w*cols); h = int(h*rows); x = int(x_cen*cols - (w/2)) ;y = int(y_cen*rows - (h/2)) ;
            else:
                x,y,w,h = None, None, None, None
            #print("frame {} yolo read ----> class {},  x_cen {},  y_cen {},  w {},  h {}".format(frame_no, class_obj, x_cen, y_cen, w, h))

    except:
        print("Cannot open file --> {}".format(location_of_yolo_truth_file))
        x,y,w,h = None, None, None, None

    return (x,y,w,h)

new_size = 672*2
dataset_672_location = "D:\\Sequence Data Feb\\seq 1 daly boh vs dundalk\\672"

rows, cols = (4320,7680)

for no in range(3,4):
    #loc = "D:\\Sequence Data Feb\\Seq 5 shelly 020\\{}\\".format(no)
    loc = "D:\\Sequence Data Feb\\seq 1 daly boh vs dundalk\\"
    frames_location = loc + "frames"
    yolo_ground_truth_location = loc + "yolo format ball"
    ground_truth_files = set(os.listdir(yolo_ground_truth_location))
    images = os.listdir(frames_location)

    for image_name in images:
        txt_file_name = image_name.split('.')[0]+'.txt'
        if txt_file_name in ground_truth_files:
            image = cv2.imread(os.path.join(frames_location,image_name))
            x,y,w,h = extract_xywh(os.path.join(yolo_ground_truth_location,txt_file_name))
            if x!=None:     #Ball is present
                new_x, new_y = extract_672_xy((x,y,w,h))
                new_bbx = x + (-1)*new_x; new_bby = y + (-1)*new_y;
                write_text = "0 {} {} {} {}".format( ((new_bbx + (w/2))/new_size), ((new_bby + (h/2))/new_size) , w/new_size, h/new_size)
                #new_x, new_y = int(random.random()*(cols-new_size)), 700
            else:       #No bounding box / No ball
                new_x, new_y = int(random.random()*(cols-new_size)), 700
                new_bbx = None; new_bby = None
                write_text = ""

            extract_672_im = image[new_y:new_y+new_size, new_x:new_x+new_size, :]

            #Write to new dataset
            cv2.imwrite( dataset_672_location+"\\"+ image_name.split('.')[0]+'.png',extract_672_im)
            with open(os.path.join(dataset_672_location,txt_file_name),'w') as f:
                f.write(write_text)

            cv2.imshow("Full Image", resize(image,500))
            if new_bbx != None:
                cv2.rectangle(extract_672_im, (new_bbx,new_bby),(new_bbx+w,new_bby+h),(0,0,255),2)
            cv2.imshow("Extracted 672 image", extract_672_im)
            key = cv2.waitKey(1) & 0xFF
            if key==ord('q'):
                break


cv2.destroyAllWindows()