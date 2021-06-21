# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:32:00 2021

Script to mark bounding box around 5 tracked objects (from ground truth) and displaying them

@author: panghals
"""

import cv2
import imutils
from stich_algo_2 import find_start_no

def resize(im,h):
    return imutils.resize(im,height = h)

if __name__ == "__main__":

    #-------------------------------------------SPECIFY MANUALLY---------------------------------------------------------
    location = "D:\\Sequence Data Feb\\Seq 5 shelly 020\\3\\"
    #------------------------------------------------------------------------------------------------------------------
    start_frame_no = find_start_no(location+"frames")[0]
    end_frame_no = start_frame_no + 490

    img_loc = location + "frames\\"
    flag = False


    p = []

    for i in range(1,12):
        p.append(open(location + "p{}.csv".format(i)))

    for i in range(1,12):
        p.append(open(location + "p_{}.csv".format(i)))

    #op_no =
    #n = op_no
    for i in range(1,23):
        p.append(open(location + "op_{}.csv".format(i)))

    #p.append(open(location + "bally_new.csv"))

    try:
        shapey = cv2.imread(img_loc+"frame_{}.tiff".format(start_frame_no)).shape
    except:
        shapey = (4320, 8192, 3)

    for pp in p:
        line = pp.readline()

    reject = []
    # rej = open(location+"reject.txt",'w')

    line = line.split(',')
    print(line[0], line[1], line[2], line[3], line[4])
    while line:
        initial_image = True
        for obj in p:
            line = obj.readline()
            if not line:
                break
            line = line.split(',')
            frame_no,x,y,w,h = int(line[0]),int(line[1]),int(line[2]),int(line[3]),int(line[4])
            if initial_image:
                image = cv2.imread(img_loc+"frame_{}.tiff".format(frame_no))
                rows,cols = image.shape[:2]
                initial_image = False

            try:
                index = p.index(obj)
            except:
                index = 12

            if index==12:
                color = (0,255,0)
            else:
                color = (0,0,255) if index>10 else (255,0,0)
            if x>=0 and x+w<shapey[1] and y>=0 and y+h<shapey[0]:
                image = cv2.rectangle(image,(x,y),(x+w,y+h),color,5)

        print("Frame No {}".format(frame_no))
        cv2.imshow("Tracking shot preview",resize(image,720))
        try:
            player_im = image[y:min(y+h,rows),x:min(x+w,cols),:]
            cv2.imshow("player", player_im)
        except:
            pass
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
        if key == ord('s'):
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key== ord('c'):
                    break
        if key == ord('r'):
            print("Frame No Rejected - {}".format(frame_no))
            # rej.write(str(frame_no)+'\n')



    for pp in p:
        pp.close()
    # rej.close()

    cv2.destroyAllWindows()