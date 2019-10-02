import cv2
from video import CaptureAsync, LandmarkDetector
from models import LandmarksFace
from misc.helpers import normalize

def main():
    cam = CaptureAsync()
    cam.start()
    detector = LandmarkDetector("./models/landmarks.dat")
    while True:
        ret, frame = cam.read()

        # Extract the bounding box and landmarks from the image
        shapes = detector.getShapes(frame)
        # Draw the bounding box and landmarks on the frame
        detected = detector.drawOverlay(frame, shapes=shapes)

        if len(shapes[0]) > 0 and len(shapes[1]) > 0:
            # Normalize distances
            normalized_shapes = normalize(shapes)
            # create face properties 
            face = LandmarksFace(normalized_shapes)
            print(face.getEyesDistance())

        cv2.imshow('App', detected)

        # show the output image with the face detections + facial landmarks
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.stop()
            break
if __name__ == "__main__":
    main()