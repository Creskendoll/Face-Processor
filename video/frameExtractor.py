import cv2
from imutils.video import FPS
from io import BytesIO
import os
import sys

video_path = sys.argv[1]
if not os.path.isfile(video_path):
    print("ERR: File %s can't be found." % video_path)
    quit()

video_folder = os.path.splitext(video_path)[0] + "/"
out_folder = "./frames/"
video = cv2.VideoCapture(video_path)

# Calculate the number of frames per second.
fps = FPS().start()

df = None

skipFrames = 50

frameIndex = 0 

if not os.path.isdir(out_folder + video_folder):
    print("Creating folder %s" % out_folder + video_folder)
    os.mkdir(out_folder + video_folder)

while True:    
    retval, frame = video.read()
    
    if not retval:
        break

    fps.update()
    frameIndex += 1 

    if frameIndex % skipFrames == 0:
        frame_name = out_folder + video_folder + "frame-" + str(frameIndex) + ".jpg" 
        print("Saving frame to %s" % frame_name)
        cv2.imwrite(frame_name, frame)

video.release()
    