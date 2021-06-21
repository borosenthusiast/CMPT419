import cv2.cv2 as cv2
import tkinter as tki
from tkinter import font as tkFont
import PIL
from PIL import Image
from PIL import ImageTk
import datetime
import os
import threading
import imutils
import sys
import subprocess
from sys import platform
import glob
import pandas as pd
import numpy as np
# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/openpose/build/python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/openpose/build/x64/Release;' +  dir_path + '/openpose/build/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('openpose/build/python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

#fps_time = 0
params = dict()
params["model_folder"] = "openpose/models"
params["write_json"] = "openpose_data"

#Starting openPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


width, height = 640, 480
cap = cv2.VideoCapture('testv.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
global_counter = 0
global_recording = False

openpose_data_path = "openpose_data/"
dataset_openpose = pd.DataFrame()

def record():
    files = glob.glob("frames/*")
    for f in files:
        os.remove(f)
    global global_recording
    global global_counter
    global_recording = True
    global_counter = 0
    print("Starting Recording!")

def predict_model():
    print("Code stub for sending data into model")

root = tki.Tk()
root.title("Emotion-Music Pair Tool")
root.bind('<Escape>', lambda e: root.quit())
largeFont = tkFont.Font(family="Helvetica", size=18, weight=tkFont.BOLD)
mediumFont = tkFont.Font(family="Helvetica", size=12)
introlbl = tki.Label(root, text="Live Video Feed", font=largeFont)
introlbl.grid(column=0, row=0)
introlbl2 = tki.Label(root, text="Processed Data", font=largeFont)
introlbl2.grid(column=2, row=0)
processed_data = PIL.ImageTk.PhotoImage(Image.open("no_data.jpg"))
lmain = tki.Label(root)
lmain.grid(column=0, row=1)
lmain2 = tki.Label(root, image = processed_data)
lmain2.grid(column=2, row=1)
recbutton = tki.Button(root, text="Determine Your Current Mood", command=record, font=largeFont)
recbutton.grid(column=0, row=2, columnspan=3 ,sticky="nesw")
helptext = tki.Text(root, font=mediumFont, wrap=tki.WORD)
helptext.insert(tki.INSERT, "The left is the live feed from the video input device on this computer. The right will populate with a processed image preview of the data gathered from the webcam. To use this program, simply click on the button and it will record a 200 frame video. Processing time will depend on your hardware configuration.\n")
helptext.insert(tki.INSERT, "Click on the button below to generate a piece of music that fits your mood!\n")
helptext.grid(column=0, row=3, columnspan=3, sticky="ew")
playbutton = tki.Button(root, text="Generate Music", font=largeFont, command=predict_model)
playbutton.grid(column=0,row=4, columnspan=3, sticky="ew")

def update_preview_image():
    imgtk = PIL.ImageTk.PhotoImage(Image.blend(Image.open("openfaceoutput/0.jpg"), Image.open("data.jpg"), 0.5)) #Creates a blend of the openface and openpose processed data
    #imgtk = PIL.ImageTk.PhotoImage(Image.open("data.jpg"))
    lmain2.configure(image=imgtk)
    lmain2.photo_ref = imgtk

def t_subproc():
    subprocess.run(['openface/FaceLandMarkImg.exe', '-fdir', 'frames', '-out_dir', 'openfaceoutput'])
    update_preview_image()
    print("Tracking Done!")

def show_video():
    global global_counter
    global global_recording
    global opWrapper #By default, uses model body_25
    _, frame = cap.read()
    #raw = frame
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    if global_recording:
        cv2.imwrite('frames/' + str(global_counter) + '.jpg', frame)
        global_counter += 1
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        print(str(datum.poseKeypoints))
        if global_counter == 199:
            newImage = datum.cvOutputData[:,:,:]
            cv2.imwrite('data.jpg', newImage)
        if global_counter >= 200:
            #t = threading.Thread(target=t_subproc)
            #t.daemon = True
            #t.start()
            global_recording = False
            #imagePaths = op.get_images_on_directory("frames");
                #cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
                #key = cv2.waitKey(15)

    imgtk = PIL.ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_video)

show_video()
root.mainloop()
