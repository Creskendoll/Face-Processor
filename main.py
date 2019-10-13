import cv2
from video import CaptureAsync, LandmarkDetector, Emotion
from models import LandmarksFace, FaceOutline
from misc.helpers import normalize
import numpy as np
from misc import Recorder
from misc import Reader
from os.path import abspath

def main():
    cam = CaptureAsync()
    emotion = Emotion()
    cam.start()
    detector = LandmarkDetector("./models/landmarks.dat")

    save_file = abspath("./recording.csv")
    recorder = Recorder(save_file)
    recording = False
    frame_index = 0

    while True:
        frame_index += 1
        ret, frame = cam.read()
        frame_H, frame_W = frame.shape[0], frame.shape[1]

        # Extract the bounding box and landmarks from the image
        shapes = detector.getShapes(frame)
        # Draw the bounding box and landmarks on the frame

        # Black background
        blank_image = np.zeros((frame_H,frame_W, 3), np.uint8)
        # Replace blank_image with frame to show the real background 
        detected = detector.drawOverlay(frame, shapes=shapes)
       
        # White background
        outlined = np.empty((frame_H, frame_W, 3), np.uint8)
        outlined.fill(255)

        if frame_index % 30 == 0:
            frame_index = 0
            # TODO: make this async
            print(emotion.getPrediction(frame))
            
        # If a face is detected in frame
        if len(shapes[0]) > 0 and len(shapes[1]) > 0:
            # Normalize distances
            normalized_shapes = normalize(shapes)
            # Values must be zipped together because of a stupid design mistake
            # Too lazy to go back and fix it
            norm_faces = zip(normalized_shapes[0], normalized_shapes[1])
            for i, norm_face in enumerate(norm_faces):
                # create face properties 
                face = LandmarksFace(norm_face)
                # print("Face #{}:\n{}".format(i, face))
                if recording:
                    # recorder.captureFrame(frame);
                    recorder.captureFace(face, i)

            faces = zip(shapes[0], shapes[1])
            for face in faces:
                # Outline drawer
                face_outline = FaceOutline(face)
                outlined = face_outline.drawOutline(outlined)
        
        frame_with_outline = np.vstack((detected, outlined))

        cv2.imshow('App', frame_with_outline)

        key = cv2.waitKey(1) & 0xFF 
        # show the output image with the face detections + facial landmarks
        if key == ord('q'):
            cam.stop()
            cv2.destroyAllWindows()
            break
        elif key == ord('s'):
            if recording:
                recorder.end()
            recording = not recording

if __name__ == "__main__":
    main()