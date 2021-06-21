
import os
import cv2
import imutils

def resize(im,h):
    return imutils.resize(im,height = h)

#-------------------------------------------SPECIFY MANUALLY---------------------------------------------------------
location = "D:\\Sequence Data Feb\\Seq 5 shelly 020\\1\\"
#------------------------------------------------------------------------
reject = location+"reject.txt"
yolo_loc = location + "yolo format ball\\"

removed = 0

with open(reject) as rej:
    fr_no = rej.readline()
    while fr_no:
        fr_no = int(fr_no)
        try:
            os.remove(yolo_loc+"frame_{}.txt".format(fr_no))
        except:
            print("MIssin frame no --> {}".format(fr_no))
        fr_no = rej.readline()
        removed += 1

print("{} files removed".format(removed))