from math import hypot

class LandmarksFace(object):
    def __init__(self, features):
        self.landmarks, self.bbox, = features
        # Map feature values to methods
        self.funcDict = {
            "eyes_distance": self.getEyesDistance,
            "eyebrow_height": self.getEyebrowHeight,
            "nose_width": self.getNoseWidth,
            "face_width": self.getFaceWidth,
            "nose_height": self.getNoseHeight,
            "mouth_width": self.getMouthWidth,
            "eye_aspect_ratio": self.getEyeAspectRatio,
            "eye_width": self.getEyeWidth,
            "eye_height": self.getEyeHeight}
        # Feature indexes
        self.left_eye_index, self.right_eye_index = 36, 45

        self.left_brow_index, self.right_brow_index, self.bottom_chin_index = 19, 24, 8

        self.left_nose_index, self.right_nose_index = 31, 35

        self.left_face_index, self.right_face_index = 0, 16

        self.nose_top_index, self.nose_bottom_index = 27, 33

        self.mouth_left_index, self.mouth_right_index = 48, 54

        # Eyes

        self.left_left, self.left_right, self.right_left, self.right_right = 36, 39, 42, 45

        self.left_top_left, self.left_top_right, self.left_bot_left, self.left_bot_right = 37, 38, 41, 40

        self.right_top_left, self.right_top_right, self.right_bot_left, self.right_bot_right = 43, 44, 47, 46

    # returns desired features in funcDict in a list (generator)
    def get(self, feature_labels):
        for feat in feature_labels:
            if feat in self.funcDict:
                # call function in funcDict
                yield self.funcDict[feat]()

    # get distance between 2 landmarks
    # takes index values of the landmarks
    def dist(self, indexA, indexB):
        landmarkA, landmarkB = self.landmarks[indexA], self.landmarks[indexB]
        return hypot(landmarkB[0] - landmarkA[0], landmarkB[1] - landmarkA[1])

    def getEyesDistance(self):
        return self.dist(self.left_eye_index, self.right_eye_index)

    def getEyebrowHeight(self):
        return (self.dist(self.left_brow_index, self.bottom_chin_index) + self.dist(self.right_brow_index, self.bottom_chin_index)) / 2

    def getNoseWidth(self):
        return self.dist(self.left_nose_index, self.right_nose_index)

    def getFaceWidth(self):
        return self.dist(self.left_face_index, self.right_face_index)

    def getNoseHeight(self):
        return self.dist(self.nose_top_index, self.nose_bottom_index)

    def getMouthWidth(self):
        return self.dist(self.mouth_left_index, self.mouth_right_index)

    def getEyeWidth(self):
        left_eye_w, right_eye_w = self.dist(
            self.left_left, self.left_right), self.dist(self.right_left, self.right_right)
        # return max(left_eye_w, right_eye_w)
        return (left_eye_w + right_eye_w) / 2

    def getEyeHeight(self):
        #left_eye_h, right_eye_h = max(self.dist(left_top_left, left_bot_left), self.dist(left_top_right, left_bot_right)), \
        #                          max(self.dist(right_top_left, right_bot_left), self.dist(right_top_right, right_bot_right))
        # return max(left_eye_h, right_eye_h)
        return (self.dist(self.left_top_right, self.left_bot_right) + self.dist(self.right_top_right, self.right_bot_right)) / 2

    def getEyeAspectRatio(self):
        return self.getEyeWidth() / self.getEyeHeight()

    def __str__(self):
        results = self.get(self.funcDict.keys())
        s = ""
        for res, key in zip(results, self.funcDict.keys()):
            s += "{}: {}\n".format(key, res)
        return s
