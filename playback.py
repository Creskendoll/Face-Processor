from misc import Reader
from os.path import abspath
import numpy as np
from time import sleep
from models.LandmarksFace import LandmarksFace
from gui.ImageBuilder import ImageBuilder
import cv2
from random import randint


def createSubsamples(data, n):
    newData = []
    for i, face in enumerate(data):
        if i < len(data) - 1:
            next_face = data[i + 1]
            current_landmarks = face.landmarks
            next_landmarks = next_face.landmarks

            l_sub_samples = [[] for _ in range(n)]
            for cur_landmark, next_landmark in zip(current_landmarks, next_landmarks):
                c_x, c_y = cur_landmark
                n_x, n_y = next_landmark
                x_diff = n_x - c_x
                y_diff = n_y - c_y
                x_interval = x_diff / n
                y_interval = y_diff / n
                samples = [
                    (c_x + (x_interval * (m - 1)), c_y + (y_interval * (m - 1)))
                    for m in range(1, n + 1)
                ]
                for sample_index in range(n):
                    l_sub_samples[sample_index].append(samples[sample_index])

            cur_bbox = face.bbox
            next_bbox = next_face.bbox
            c_x, c_y = cur_bbox[0], cur_bbox[1]
            n_x, n_y = next_bbox[0], next_bbox[1]
            size_diff = cur_bbox[2] + next_bbox[2]
            x_diff = n_x - c_x
            y_diff = n_y - c_y
            x_interval = x_diff / n
            y_interval = y_diff / n
            b_sub_samples = [
                (
                    c_x + (x_interval * (m - 1)),
                    c_y + (y_interval * (m - 1)),
                    size_diff / 2,
                    size_diff / 2,
                )
                for m in range(1, n + 1)
            ]

            for n_l, n_b in zip(l_sub_samples, b_sub_samples):
                features = (n_l, n_b)
                newData.append(LandmarksFace(features))

    return newData


def unNormalize(face, dim):
    w, h = dim
    l, b = face.landmarks, face.bbox
    b_n = (int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h))
    offset_x = b_n[0]
    offset_y = b_n[1]
    b_size = b_n[2]
    l_n = [
        [(int((x * b_size) + offset_x), int((y * b_size) + offset_y)) for (x, y) in l]
    ]
    return (l_n, [b_n])


file = abspath("./recording.csv")
model_file = abspath("./models/landmarks.dat")
reader = Reader(file)
img_builder = ImageBuilder(model_file, file)
data = []

for row in reader.read_csv():
    landmarks, bbox, face_index = row[:-2], eval(row[-2]), eval(row[-1])
    landmarks = [eval(l) for l in landmarks]
    features = (landmarks, bbox)
    face = LandmarksFace(features)
    data.append(face)

FPS = 200
FRAME_HEIGHT, FRAME_WIDTH = 960, 1280
TRAIL = 6
TRAIL_MARGIN = 3

frame_index = 0

shapes_buffer = []
more_data = createSubsamples(data, 15)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_writer = cv2.VideoWriter("output.avi", fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

for frame_info in more_data:
    # White background
    bg = np.empty((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    bg.fill(0)

    shapes = unNormalize(frame_info, (FRAME_WIDTH, FRAME_HEIGHT))

    drawn_img = img_builder.build(
        bg, draw_landmarks=True, draw_outline=False, shapes=shapes
    )

    if frame_index % TRAIL_MARGIN == 0:
        frame_index = 0
        shapes_buffer.append(shapes)

    prev_face = None
    for past_shape in shapes_buffer:
        drawn_img = img_builder.build(
            drawn_img, draw_landmarks=True, draw_outline=False, shapes=past_shape
        )

        # if prev_face is not None:
        #     prev_landmarks = prev_face[0][0]
        #     current_landmarks = past_shape[0][0]
        #     for point_prev, point_current in zip(prev_landmarks, current_landmarks):
        #       x1, y1 = point_prev
        #       x2, y2 = point_current
        #       rand_color = (randint(0,255),randint(0,255),randint(0,255))
        #       cv2.line(drawn_img, (x1,y1), (x2,y2), rand_color, thickness=2)
        # prev_face = past_shape

    cv2.imshow("Playback", drawn_img)
    # video_writer.write(drawn_img)
    if len(shapes_buffer) == TRAIL:
        shapes_buffer.pop(0)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    frame_index += 1
    sleep(1.0 / FPS)

video_writer.release()
cv2.destroyAllWindows()
