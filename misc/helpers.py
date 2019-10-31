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

def shapeToData(shapes):
    if len(shapes[0]) > 0 and len(shapes[1]) > 0:
        # Normalize the values
        normalized_shapes = normalize(shapes)
        # Values must be zipped together because of a stupid design mistake
        # Too lazy to go back and fix it
        # faces, norm_faces
        return zip(shapes[0], shapes[1]), zip(normalized_shapes[0], normalized_shapes[1])
    else:
        return None
