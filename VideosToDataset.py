import cv2
import ffmpeg
import os
from pathlib import Path

def checkVideoRotation(videoPath):
    metadata = ffmpeg.probe(videoPath)

    if int(metadata['streams'][0]['tags']['rotate']) == 90:
        return cv2.ROTATE_90_CLOCKWISE
        
    elif int(metadata['streams'][0]['tags']['rotate']) == 180:
        return cv2.ROTATE_180
        
    elif int(metadata['streams'][0]['tags']['rotate']) == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE

    return None

def convertVideoToImages(videoPath, targetPath, videoName):
    cap = cv2.VideoCapture(videoPath)
    rotateCode = checkVideoRotation(videoPath)
    
    i = 0
    
    if not cap.isOpened():
        print("Error while opening the video:", videoPath)
        
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            frameName = targetPath + videoName + "_" + str(i) + ".jpg"
            
            if rotateCode is not None:
                frame = cv2.rotate(frame, rotateCode)
            
            cv2.imwrite(frameName, frame)
            i += 1
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()



VideosPath = "./Videos/Train"
DatasetPath = "./Dataset/Train"

Categories = next(os.walk(VideosPath))[1]

for category in Categories:
    categoryPath = VideosPath + category
    
    categoryVideos = next(os.walk(categoryPath))[2]
    
    for video in categoryVideos:
        videoPath = categoryPath + "/" + video
        targetPath = DatasetPath + category + "/"
        
        Path(targetPath).mkdir(parents = True, exist_ok = True)
        convertVideoToImages(videoPath, targetPath, video)
        print(video, "Done!")
    

