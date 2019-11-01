from video import CaptureAsync, LandmarkDetector, Emotion
from models import LandmarksFace, FaceOutline
from misc.helpers import normalize, shapeToData
import numpy as np
from misc import Recorder, Reader, Plotter
import cv2

class ImageBuilder(object):
    def __init__(self, model_file, save_file):
        self.emotion = Emotion()
        self.detector = LandmarkDetector(model_file)

        self.recorder = Recorder(save_file)
        self.recording = False

    def getFrameData(self, frame, shapes=None):
        # Extract the bounding boxes and landmarks from the image
        shapes = self.detector.getShapes(frame)
        if len(shapes[0]) > 0 and len(shapes[1]) > 0:
            # Normalize the values
            normalized_shapes = normalize(shapes)
            # Values must be zipped together because of a stupid design mistake
            # Too lazy to go back and fix it
            # faces, norm_faces
            return zip(shapes[0], shapes[1]), zip(normalized_shapes[0], normalized_shapes[1])
            # faces and norm_faces contain the bounding boxes and landmarks required to construct Face objects
            # i represents the face index since there can be multiple faces in the frame
        else:
            return None

    def getFaceImages(self, frame, data=None, size=(128,128)):
        if data is None:
            data = self.getFrameData(frame)

        if data is not None:
            faces, _ = data
            face_imgs = [LandmarksFace(face).getFaceImage(frame, size) for face in faces]

            return [f for f in face_imgs if f is not None]
        else:
            return []

    def build(self, frame, draw_landmarks=True, draw_outline=True, shapes=None):
        if not draw_landmarks and not draw_outline:
            return frame

        frame_H, frame_W = frame.shape[0], frame.shape[1]
        
        if shapes is None:
            # Extract the bounding boxes and landmarks from the image
            shapes = self.detector.getShapes(frame)

        data = shapeToData(shapes)
        # Black background
        # blank_image = np.zeros((frame_H, frame_W, 3), np.uint8)
        if draw_landmarks:
            # Draw the bounding box and landmarks on the frame
            frame = self.detector.drawOverlay(frame, shapes=shapes)

        if data is not None:
            faces, norm_faces = data
            # faces and norm_faces contain the bounding boxes and landmarks required to construct Face objects
            # i represents the face index since there can be multiple faces in the frame
            for i, (face, norm_face) in enumerate(zip(faces, norm_faces)):
                # create face properties
                landmarks_face = LandmarksFace(norm_face)
                # print("Face #{}:\n{}".format(i, face))
                # Record landmarks
                if self.recording:
                    # recorder.captureFrame(frame);
                    self.recorder.captureFace(landmarks_face, i)

                # Outline drawer
                face_outline = FaceOutline(face)
                if draw_outline:
                    frame = face_outline.drawOutline(frame)

        return frame

    def toggleRecording(self):
        if self.recording:
            print("Saving to:", self.recorder.file_path)
            self.recorder.end()
        else:
            print("Started recording.")
        self.recording = not self.recording