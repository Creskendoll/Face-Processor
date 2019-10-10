import cv2


class Recorder(object):
    def __init__(self, file_path):
        self.file_path = filepath
        self.data = []
        self.video_out = cv2.VideoWriter(
            'output.avi', -1, 20.0, (640, 480))

    def captureFrame(self, frame):
        self.out.write(frame)

    def captureFace(self, face):
        self.data.append(face.landmarks)

    def end(self):
        self.video_out.release()
    