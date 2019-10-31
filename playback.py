from misc import Reader
from os.path import abspath
import numpy as np
from time import sleep

file = abspath("./recording.csv")
reader = Reader(file)

frames_info = []

for row in reader.read_csv():
    landmarks, face_index = row[:-1], eval(row[-1])
    landmarks = [eval(l) for l in landmarks]
    frames_info.append(landmarks)
    print("Face index: {}\nLandmarks: {}".format(face_index, landmarks))

FPS = 20.0
FRAME_HEIGHT, FRAME_WIDTH = 640, 480
for frame_info in frames_info:
        
    # White background
    outlined = np.empty((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    outlined.fill(255)
    sleep(1.0/FPS)
