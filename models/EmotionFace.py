class EmotionFace(object):
    def __init__(self, data):
        """Init a face object.
        Takes in names of the columns and the rows.

        The values are indexed with numbers rather than keywords.
        """
        # Extract the key and values from the data
        self.keys = list(data.keys())
        self.values = list(data.values())

        # To access the emotions from outside
        self.emotions = [
            "anger",
            "contempt",
            "disgust",
            "fear",
            "happiness",
            "neutral",
            "sadness",
            "surprise"]

        # Build a dictionary of indexes
        self.index_dict = {
            "anger": self.keys.index("faceAttributes_emotion_anger"),
            "contempt": self.keys.index("faceAttributes_emotion_contempt"),
            "disgust": self.keys.index("faceAttributes_emotion_disgust"),
            "fear": self.keys.index("faceAttributes_emotion_fear"),
            "happiness": self.keys.index("faceAttributes_emotion_happiness"),
            "neutral": self.keys.index("faceAttributes_emotion_neutral"),
            "sadness": self.keys.index("faceAttributes_emotion_sadness"),
            "surprise": self.keys.index("faceAttributes_emotion_surprise"),
            "rectangle_left": self.keys.index("faceRectangle_left"),
            "rectangle_top": self.keys.index("faceRectangle_top"),
            "rectangle_width": self.keys.index("faceRectangle_width"),
            "rectangle_height": self.keys.index("faceRectangle_height"),
            "gender": self.keys.index("faceAttributes_gender"),
            "smile": self.keys.index("faceAttributes_smile"),
            "age": self.keys.index("faceAttributes_age")}

    def getFacePosition(self):
        # (x1, y1, x2, y2)
        face_rectangle = (self.values[self.index_dict["rectangle_left"]], self.values[self.index_dict["rectangle_top"]],
                          self.values[self.index_dict["rectangle_left"]] +
                          self.values[self.index_dict["rectangle_width"]],
                          self.values[self.index_dict["rectangle_top"]] + self.values[self.index_dict["rectangle_height"]])

        return face_rectangle

    def getAllEmotions(self):
        """Returns all the emotion values as a dictionary.
        """

        values = [self.values[self.index_dict[e]] for e in self.emotions]

        return dict(zip(self.emotions, values))

    def getEmotion(self, emotion):
        emotion = emotion.lower()
        assert emotion in self.emotions, "Wrong key supplied for emotion! Valid keys:\n{}".format(self.emotions)
        emotions = self.getAllEmotions()        
        return emotions[emotion]

    def getGender(self):
        return self.values[self.index_dict["gender"]]

    def getAge(self):
        return self.values[self.index_dict["age"]]

    def __str__(self):
        s = "Face:\n"
        values = self.getAllEmotions()

        for key in values:
            s += "{}: {}\n".format(key, values[key])
        return s