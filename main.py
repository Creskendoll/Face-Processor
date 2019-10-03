import cv2
from video import CaptureAsync, LandmarkDetector
from models import LandmarksFace, FaceOutline
from misc.helpers import normalize
import numpy as np

def main():
    cam = CaptureAsync()
    cam.start()
    detector = LandmarkDetector("./models/landmarks.dat")
    while True:
        ret, frame = cam.read()
        frame_H, frame_W = frame.shape[0], frame.shape[1]

        # Extract the bounding box and landmarks from the image
        shapes = detector.getShapes(frame)
        # Draw the bounding box and landmarks on the frame

        # Black background
        blank_image = np.zeros((frame_H,frame_W, 3), np.uint8)
        # Replace blank_image with frame to show the real background 
        detected = detector.drawOverlay(frame, shapes=shapes)
       
        outlined = np.empty((frame_H, frame_W, 3), np.uint8)
        outlined.fill(255)

        if len(shapes[0]) > 0 and len(shapes[1]) > 0:
            # Normalize distances
            normalized_shapes = normalize(shapes)
            # create face properties 
            face = LandmarksFace(normalized_shapes)
            
            # Outline drawer
            face_outline = FaceOutline(shapes)
            outlined = face_outline.drawOutline(outlined)

            print(face)
        
        frame_with_outline = np.hstack((detected, outlined))

        cv2.imshow('App', frame_with_outline)

        # show the output image with the face detections + facial landmarks
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.stop()
            break
if __name__ == "__main__":
    main()