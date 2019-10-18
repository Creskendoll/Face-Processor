import requests
import cv2
import os
from flatten_json import flatten
import json
from io import BytesIO
import sys
from models import EmotionFace
import threading

class Emotion(object):
    # https://westus.dev.cognitive.microsoft.com/docs/services/563879b61984550e40cbbe8d/operations/563879b61984550f30395236
    def __init__(self):
        self.subscription_key = "b0c13e4f08324d04ac646dc5f5c1cf89" 
        self.emotion_recognition_url = "https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect"
        
        self.header = {'Ocp-Apim-Subscription-Key': self.subscription_key, "Content-Type": "application/octet-stream"  }

        self.params = {
            'returnFaceId': 'true',
            'returnFaceLandmarks': 'false',
            'returnFaceAttributes': 'age,gender,smile,emotion'
        }

    def postRequest(self, images, callback):
        faces = []
        for image in images:
            encoded_image = cv2.imencode(".jpg", image)[1]
            encoded_image = encoded_image.tobytes()
            resp = requests.post(self.emotion_recognition_url, params=self.params, headers=self.header, data=encoded_image)

            if not resp.status_code == 200:
                print("API Error:", resp.text)
                callback(None)

            # Flatten the response 
            # The response is an array of face data
            flat_resp = [flatten(o) for o in resp.json()]
            # Build the response data into Face objects
            faces += [EmotionFace(face_data) for face_data in flat_resp]

        callback(faces)

    def getPredictionAsync(self, image, callback):
        assert self.subscription_key != "", "Subscription key for the emotion API can not be empty!"
        
        threading.Thread(target=self.postRequest, args=(image, callback)).start()
