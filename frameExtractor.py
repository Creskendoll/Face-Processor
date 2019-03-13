import cv2
from imutils.video import FPS
from io import BytesIO
import os

video_path = "vid.mp4"
video_folder = "vid"
out_folder = "./frames/"
video = cv2.VideoCapture(video_path)

# Calculate the number of frames per second.
fps = FPS().start()

df = None

frameIndex = 0 

if not os.path.isdir(out_folder + video_folder):
    os.mkdir(out_folder + video_folder)

while True:    
    retval, frame = video.read()
    
    if not retval:
        break

    fps.update()
    frameIndex = frameIndex + 1 

    if frameIndex % 10 == 0:
        cv2.imwrite("./frames/vid/frame-" + str(frameIndex) + ".jpg", frame)

video.release()
    