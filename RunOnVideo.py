# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:39:52 2020

@author: levan
"""
import cv2
import ffmpeg
import numpy as np

from tensorflow.keras.models import load_model

def checkVideoRotation(videoPath):
    metadata = ffmpeg.probe(videoPath)

    code = None
    
    if int(metadata['streams'][0]['tags']['rotate']) == 90:
        code = cv2.ROTATE_90_CLOCKWISE
    elif int(metadata['streams'][0]['tags']['rotate']) == 180:
        code = cv2.ROTATE_180
    elif int(metadata['streams'][0]['tags']['rotate']) == 270:
        code = cv2.ROTATE_90_COUNTERCLOCKWISE

    return code

def frameForModel(frame):
    f = cv2.resize(frame, (128, 128))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64)
    f = np.expand_dims(f, axis = 0)
    f = np.expand_dims(f, axis = -1)
    f /= 255.
    
    return f
    

videoPath = './20200405_201620_1.mp4'
modelPath = './saved/saved.h5'

model = load_model(modelPath)


font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 6
fontColor              = (255, 0, 0)
lineType               = 10


count = 0
currentState = [-1, 0]
minStrength = 3
canIncreaseCount = False

cap = cv2.VideoCapture(videoPath)
# cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Frame', 480, 640)

rotateCode = checkVideoRotation(videoPath)

out = cv2.VideoWriter('output.mp4', -1, 60.0, (1080, 1920))

if not cap.isOpened():
    print("Error while opening the video:", videoPath)
    
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        if rotateCode is not None:
            frame = cv2.rotate(frame, rotateCode)
            
        pred = model(frameForModel(frame)).numpy()
        
        c = np.argmax(pred)
        
        category = 'None'
        state = -1
        
        if pred[0][c] >= 0.5:
            if c == 0:
                category = "Lower"
                state = 0
            elif c == 1:
                category = "Middle"
                state = 1
            elif c == 2:
                category = "Upper"
                state = 2
            
        if currentState[0] == state:
            currentState[1] += 1
        else:
            currentState = [state, 1]
            
        if currentState[1] > minStrength:
            if currentState[0] == 1:
                canIncreaseCount = True
            elif currentState[0] == 2 and canIncreaseCount:
                count += 1
                canIncreaseCount = False
        
        cv2.putText(frame,
                    category, 
                    (20, 1700), 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        
        cv2.putText(frame,
                    "Count: " + str(count), 
                    (20, 400), 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        
        
        # cv2.imshow('Frame', frame)
        out.write(frame)

        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()