import cv2
from video import CaptureAsync, LandmarkDetector, Emotion
from models import LandmarksFace, FaceOutline
from misc.helpers import normalize
import numpy as np
from misc import Recorder, Reader, Plotter
from os.path import abspath
from gui.UI import UI
from gui.ImageBuilder import ImageBuilder
from misc.helpers import normalize, shapeToData

p = Plotter()
def livePlotEmotions(emotion_faces: list):
    """
        `emotion_faces` is a list of EmotionFace objects.

        Updates a matplotlib plot with the emotion values found in the emotion_faces argument.
        Refer to this link for more info on live plotting: 
        https://makersportal.com/blog/2018/8/14/real-time-graphing-in-python

        There should be one chart for each face and a single chart should plot all the emotions of a single face
        Each line will represent an individual emotion, example:
        https://pydatascience.org/2017/11/24/plot-multiple-lines-in-one-chart-with-different-style-python-matplotlib/
    """
    for face in emotion_faces:
        # TODO-Kaan: Update a plt with emotion values
        p.updatePlot(face)
        print(face)

def main():
    cam = CaptureAsync()
    emotion = Emotion()
    cam.start()
    model_file = abspath("./models/landmarks.dat")
    detector = LandmarkDetector(model_file)

    save_file = abspath("./recording.csv")
    recorder = Recorder(save_file)
    recording = False
    frame_index = 0
    img_builder = ImageBuilder(model_file, save_file)
    # Change this to determine how frequently the emotions will get updated
    # The Microsoft API has a limit of how many requests can be made each minute
    # Try increasing the value if the API returns an error
    UPDATE_FREQ = 40

    while True:
        # Get key input
        key = cv2.waitKey(1) & 0xFF 

        # Read camera
        ret, frame = cam.read()
        frame_H, frame_W = frame.shape[0], frame.shape[1]

        # Extract the bounding boxes and landmarks from the image
        shapes = detector.getShapes(frame)
       
        # White background
        # outlined = np.empty((frame_H, frame_W, 3), np.uint8)
        # outlined.fill(255)

        # Frequently get predictions
        # frame_index += 1
        # if frame_index % UPDATE_FREQ == 0:
        #     frame_index = 0
        #     emotion.getPredictionAsync(frame, lambda res: print(res))

        processed_img = img_builder.build(frame, shapes=shapes)
        face_imgs = img_builder.getFaceImages(frame, data=shapeToData(shapes), size=(256,256))
        if len(face_imgs) > 0:
            cv2.imshow('Face', face_imgs[0])
        
        # Get emotions from Microsoft API when 'e' is pressed
        if key == ord("e"):
            # TODO-Kaan: Get the predictions from the face image and send EmotionFace list to the callback
            # You might wanna add the face index to the callback parameters idk.
            emotion.getPredictionAsync(face_imgs, livePlotEmotions)
        
        # Vertically stack the images
        # frame_with_outline = np.vstack((detected, outlined))
        cv2.imshow('App', processed_img)

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