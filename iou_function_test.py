# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:14:15 2021

@author: panghals
"""

def calc_iou(a,b):    #x1 y1 (top left)  x2 y2 (bottom right)
    boxA = (a[0],a[1],a[0]+a[2],a[1]+a[3]); boxB = (b[0], b[1], b[0]+b[2],b[1]+b[3])
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1]); xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3]);
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1])) ;boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


x,y,w,h = 1,0,2,2
x2,y2,w2,h2 = 2,1,2,2

print("x y w h")
print("x {} y {} w {} h {} <-> x2 {} y2 {} w2 {} h2 {}".format(x,y,w,h,x2,y2,w2,h2))
print("IOU {}".format(calc_iou((x,y,w,h),(x2,y2,w2,h2))))