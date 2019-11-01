from misc import Reader
from os.path import abspath
import numpy as np
from time import sleep
from models.LandmarksFace import LandmarksFace
from gui.ImageBuilder import ImageBuilder
import cv2

def unNormalize(face, dim):
    w,h = dim
    l, b = face.landmarks, face.bbox
    l = [[[x*w, y*h] for (x, y) in l]]
    b = [(b[0]*w,b[1]*h,b[2]*w,b[3]*h)]
    return (l, b)

file = abspath("./recording.csv")
model_file = abspath("./models/landmarks.dat")
reader = Reader(file)
img_builder = ImageBuilder(model_file, file)
frames_info = []

for row in reader.read_csv():
    landmarks, bbox, face_index = row[:-2], eval(row[-2]), eval(row[-1])
    landmarks = [eval(l) for l in landmarks]
    features = (landmarks, bbox)
    face = LandmarksFace(features)
    frames_info.append(face)

    # print("Face index: {}\nLandmarks: {}".format(face_index, landmarks))

FPS = 20.0
FRAME_HEIGHT, FRAME_WIDTH = 640, 480
for frame_info in frames_info:
    
    # White background
    bg = np.empty((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    bg.fill(255)

    shapes = unNormalize(frame_info, (FRAME_WIDTH, FRAME_HEIGHT))
    drawn_img = img_builder.build(bg, draw_landmarks=True, draw_outline=False, shapes=shapes)
    cv2.imshow('Playback', drawn_img)

    sleep(1.0/FPS)
