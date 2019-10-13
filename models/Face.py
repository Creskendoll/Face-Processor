class Face(object):
    def __init__(self, data, columns):
        """Init a face object.
        Takes in names of the columns and the rows.

        The values are indexed with numbers rather than keywords.
        """
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
            "anger": columns.get_loc("faceAttributes_emotion_anger"),
            "contempt": columns.get_loc("faceAttributes_emotion_contempt"),
            "disgust": columns.get_loc("faceAttributes_emotion_disgust"),
            "fear": columns.get_loc("faceAttributes_emotion_fear"),
            "happiness": columns.get_loc("faceAttributes_emotion_happiness"),
            "neutral": columns.get_loc("faceAttributes_emotion_neutral"),
            "sadness": columns.get_loc("faceAttributes_emotion_sadness"),
            "surprise": columns.get_loc("faceAttributes_emotion_surprise"),
            "rectangle_left": columns.get_loc("faceRectangle_left"),
            "rectangle_top": columns.get_loc("faceRectangle_top"),
            "rectangle_widht": columns.get_loc("faceRectangle_width"),
            "rectangle_height": columns.get_loc("faceRectangle_height"),
            "gender": columns.get_loc("faceAttributes_gender"),
            "smile": columns.get_loc("faceAttributes_smile"),
            "age": columns.get_loc("faceAttributes_age")}

        # Store the rows
        self.data = data

    def getFacePosition(self):
        # (x1, y1, x2, y2)
        face_rectangle = (self.data[self.index_dict["rectangle_left"]], self.data[self.index_dict["rectangle_top"]],
                          self.data[self.index_dict["rectangle_left"]] +
                          self.data[self.index_dict["rectangle_widht"]],
                          self.data[self.index_dict["rectangle_top"]] + self.data[self.index_dict["rectangle_height"]])

        return face_rectangle

    def getEmotions(self):
        """Returns all the emotion values as a dictionary.
        """

        values = [self.data[self.index_dict[e]] for e in self.emotions]

        return dict(zip(emotions, values))

    def getEmotion(self, emotion):
        emotions = self.getEmotions()
        try:
            result = emotions[emotion]
        except KeyError:
            print("Error! Wrong key supplied for emotion! Valid keys:", self.emotions)

        return result

    def getGender(self):
        return self.data[self.index_dict["gender"]]

    def getAge(self):
        return self.data[self.index_dict["age"]]

    def __str__(self):
        return self.getEmotions()