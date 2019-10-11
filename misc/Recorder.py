import cv2
import csv
from itertools import chain

class Recorder(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.video_out = cv2.VideoWriter(
            'output.avi', -1, 20.0, (640, 480))

    def captureFrame(self, frame):
        self.out.write(frame)

    def captureFace(self, face, face_index):
        self.data.append(face.landmarks + [face_index])

    def save_csv(self):
        assert ".csv" in self.file_path, "File type must be of type CSV"
        with open(self.file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

            # data = list(chain.from_iterable(self.data))
            for row in self.data:
                writer.writerow(row)

    def end(self):
        self.save_csv()
        # self.video_out.release()
    