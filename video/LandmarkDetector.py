import dlib
import cv2
from imutils import face_utils
from itertools import chain

class LandmarkDetector(object):
    def __init__(self, model_file):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_file)

    def getShapes(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = self.detector(gray, 1)
        
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        landmarks = [face_utils.shape_to_np(self.predictor(gray, rect)) for rect in rects]

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        bbox = [face_utils.rect_to_bb(rect) for rect in rects]
        
        # Flatten the list of landmarks
        flat_landmarks = list(chain.from_iterable(landmarks))
        return (flat_landmarks, bbox)

    def drawOverlay(self, img, shapes=None):
        result_img = img.copy()

        if shapes is None:
            # detect faces bounding boxes and landmarks in the image
            shapes = self.getShapes(result_img)

        # unpack landmarks and bboxes
        landmarks, rects = shapes
        for i, rect in enumerate(rects):
            # draw rectangles around faces
            (x, y, w, h) = rect
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face numbers
            cv2.putText(result_img, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw the landmarks on the faces as circles
        for (x, y) in landmarks:
            cv2.circle(result_img, (x, y), 3, (0, 0, 255), -1)

        return result_img