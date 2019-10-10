from models import LandmarksFace
import cv2

# Returns normalized [bbox, [(landmarkX, landmarkY)...]]


def normalize(features):
    # landmarks, bboxes = features
    assert len(features[0]) > 0 and len(
        features[1]) > 0, "No faces are provided to the normalize function!"
    
    faces = zip(features[0], features[1])
    norm_landmarks = []
    for face in faces:
        landmarks, rect = face
        faceX, faceY, faceW, faceH = rect
        # don't ask me how
        norm_landmarks.append([((x - faceX) / faceW, (y - faceY) / faceH)
                               for (x, y) in landmarks])

    return (norm_landmarks, features[1])


# Returns a collection of bounding boxes and landmarks of the faces in the image
# [(bounding_box, [(x,y), ...])]
def getFeatures(img):
    result_img = img.copy()
    # Images are initially read in BGR color space
    gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        bbox = face_utils.rect_to_bb(rect)
        yield (bbox, [dot for dot in shape])


def drawFeatures(img):

    result_img = img.copy()
    face_feats = list(getFeatures(result_img))
    face = LandmarksFace(face_feats[0])

    for (bbox, landmarks) in face_feats:
        (x, y, w, h) = bbox
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0, 255), 4)

        for i, (dot_x, dot_y) in enumerate(landmarks):
            #cv2.putText(result_img,str(i),(dot_x, dot_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4, 255)
            cv2.circle(result_img, (dot_x, dot_y), 2, (255, 0, 0, 255), -1)

    # TODO
    # mouth width
    pointA_X, pointA_Y = face.landmarks[face.mouth_left_index]
    pointB_X, pointB_Y = face.landmarks[face.mouth_right_index]
    dist = face.dist(face.mouth_right_index, face.mouth_left_index)
    cv2.line(result_img, (pointA_X, pointA_Y),
             (pointB_X, pointB_Y), (0, 0, 255, 255), 2)
    cv2.putText(result_img, "Mouth Width", (int(pointA_X+dist/2),
                                            pointA_Y-10), cv2.FONT_HERSHEY_DUPLEX, 0.65, 255)

    # Nose width
    pointA_X, pointA_Y = face.landmarks[face.left_nose_index]
    pointB_X, pointB_Y = face.landmarks[face.right_nose_index]
    dist = face.dist(face.left_nose_index, face.right_nose_index)
    cv2.line(result_img, (pointA_X, pointA_Y),
             (pointB_X, pointB_Y), (0, 0, 255, 255), 2)
    cv2.putText(result_img, "Nose Width", (int(pointA_X+dist/2),
                                           pointA_Y-10), cv2.FONT_HERSHEY_DUPLEX, 0.65, 255)

    # Face width
    pointA_X, pointA_Y = face.landmarks[face.left_face_index]
    pointB_X, pointB_Y = face.landmarks[face.right_face_index]
    dist = face.dist(face.left_face_index, face.right_face_index)
    cv2.line(result_img, (pointA_X, pointA_Y),
             (pointB_X, pointB_Y), (0, 0, 255, 255), 2)
    cv2.putText(result_img, "Face Width", (int(pointA_X+dist/2),
                                           pointA_Y-15), cv2.FONT_HERSHEY_DUPLEX, 0.65, 255)

    return result_img
