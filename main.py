import cv2
from video import CaptureAsync, LandmarkDetector

def main():
    cam = CaptureAsync()
    cam.start()
    detector = LandmarkDetector("./models/landmarks.dat")
    while True:
        ret, frame = cam.read()

        detected = detector.drawOverlay(frame)
        cv2.imshow('App', detected)

        # show the output image with the face detections + facial landmarks
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.stop()
            break
if __name__ == "__main__":
    main()