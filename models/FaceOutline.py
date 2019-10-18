from models import LandmarksFace
import cv2

class FaceOutline(LandmarksFace):
    # Features should be of absolute unit
    # Normalized values won't work since the outline will be drawn on a
    # equally big surface
    def __init__(self, features):
        self.landmarks, self.bbox, = features

        self.chin_nodes = list(range(17))
        self.mouth_nodes = list(range(48,68))
        self.left_eyebrow_nodes = list(range(17,22))
        self.left_eye_nodes = list(range(36,42))
        self.right_eyebrow_nodes = list(range(22, 27)) 
        self.right_eye_nodes = list(range(42, 48))
        self.nose_nodes = list(range(27, 36))

    def drawOutline(self, img):
        result = img.copy()
        self.drawEyes(result)
        self.drawChin(result)
        self.drawMouth(result)
        self.drawNose(result)
        return result

    def drawChin(self, img):
        for i, chin_node in enumerate(self.chin_nodes[:-1]):
            x1, y1 = self.landmarks[chin_node]
            x2, y2 = self.landmarks[chin_node+1]
            cv2.line(img, (x1,y1), (x2,y2), (0,0,0), thickness=2)

    def drawMouth(self, img):
        for i, mouth_node in enumerate(self.mouth_nodes[:-1]):
            x1, y1 = self.landmarks[mouth_node]
            x2, y2 = self.landmarks[mouth_node+1]
            cv2.line(img, (x1,y1), (x2,y2), (0,0,0), thickness=2)
    
    def drawNose(self, img):
        for i, nose_node in enumerate(self.nose_nodes[:-1]):
            x1, y1 = self.landmarks[nose_node]
            x2, y2 = self.landmarks[nose_node+1]
            cv2.line(img, (x1,y1), (x2,y2), (0,0,0), thickness=2)

    # takes in the reference of an image
    # the function will override the image
    def drawEyes(self, img):
        for i, eye_node in enumerate(self.left_eye_nodes[:-1]):
            x1, y1 = self.landmarks[eye_node]
            x2, y2 = self.landmarks[eye_node+1]
            cv2.line(img, (x1,y1), (x2,y2), (0,0,0), thickness=2)
        for i, eye_node in enumerate(self.left_eyebrow_nodes[:-1]):
            x1, y1 = self.landmarks[eye_node]
            x2, y2 = self.landmarks[eye_node+1]
            cv2.line(img, (x1,y1), (x2,y2), (0,0,0), thickness=2)
        for i, eye_node in enumerate(self.right_eyebrow_nodes[:-1]):
            x1, y1 = self.landmarks[eye_node]
            x2, y2 = self.landmarks[eye_node+1]
            cv2.line(img, (x1,y1), (x2,y2), (0,0,0), thickness=2)
        for i, eye_node in enumerate(self.right_eye_nodes[:-1]):
            x1, y1 = self.landmarks[eye_node]
            x2, y2 = self.landmarks[eye_node+1]
            cv2.line(img, (x1,y1), (x2,y2), (0,0,0), thickness=2)
