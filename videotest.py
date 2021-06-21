import cv2.cv2 as cv2
import numpy as np
import pandas as pd
import os

def video_capture(fname):
    vid = cv2.VideoCapture(0)
    if (vid.isOpened() == False):
        print("Unable to open or access camera.")
   # count = 0
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    while (True):
        ret, frame = vid.read()
        if ret == True:
            out.write(frame)
            #count += 1
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
            
    vid.release()
    out.release()
    cv2.destroyAllWindows()

#def openpose_process():
    

video_capture('file.mp4')