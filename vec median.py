#------------------VECTOR MEDIAN BACKGROUND PLATE GENERATION (MULTI THREADING) -------------------

import logging
import threading
import cv2
import os
import numpy as np

loc = "D:\\Sequence Data Feb\\Seq 5 shelly 020\\extra frames 2\\"
frames_720 = loc + "1080\\"
write_loc =  "D:\\Sequence Data Feb\\Seq 5 shelly 020\\vec_median_1080.png";
files = os.listdir(frames_720)
temp_image = cv2.imread(frames_720 + files[0])
vec_median = np.ones(temp_image.shape,temp_image.dtype)
col_start = 0; col_end = temp_image.shape[1];


start = 0
frames = []
for i in range(50):
    frames.append(cv2.imread(frames_720+files[i*10]))
    
def eucl_dist(pix1,pix2):
    return ((np.sum(cv2.absdiff(pix1,pix2).astype("uint16")**2))**0.5).astype("uint8")

def calculate_vec_median(thread_no, row_start, row_end):
    
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            
            print("Thread {}\t {} % complete \n".format(thread_no, (100)*((row-row_start)*(col_end)+col) /(108*col_end)))
            
            cost_mat = [[0 for _ in range(50)] for __ in range(50)]
            for i in range(50):
                for j in range(i+1,50):
                    cost_mat[i][j] = eucl_dist(frames[i][row,col,:],frames[j][row,col,:])
                    cost_mat[j][i] = cost_mat[i][j]
                    
            #Check which frames' pixel had mini cost
            mini_i = 0; min_cost = sum(cost_mat[0])
            for i in range(50):
                costy = sum(cost_mat[i])
                if costy < min_cost:
                    mini_i = i
                    min_cost = costy
                    
            #Use that frames' pixel at that location in vec median
            vec_median[row,col,:] = frames[mini_i][row,col,:]
            
row_start = 0

threads = list()
for index in range(10):
    logging.info("Main    : create and start thread %d.", index)
    thread = threading.Thread(target=calculate_vec_median, args=(index,row_start,row_start+108))
    threads.append(thread)
    thread.start()
    row_start+=108

for index, thread in enumerate(threads):
    
    logging.info("Main    : before joining thread %d.", index)
    thread.join()
    logging.info("Main    : thread %d done", index)
    
cv2.imwrite(write_loc,vec_median)
    
#cv2.imshow("vector median", vec_median)
#cv2.waitKey(0)