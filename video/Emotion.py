import requests
import cv2
import os
from flatten_json import flatten
import json
from io import BytesIO
import sys
from models import Face

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

    def getPrediction(self, image):
        assert self.subscription_key != "", "ERR: subscription_key can not be empty! Check line 10 in index.py"
        
        encoded_frame = cv2.imencode(".jpg", image)[1]
        encoded_frame = encoded_frame.tobytes()
        resp = requests.post(self.emotion_recognition_url, params=self.params, headers=self.header, data=encoded_frame)

        if not resp.status_code == 200:
            print("ERR: Bad request:", resp.text)
            return None

        # Flatten the response 
        # resp = (flatten(o) for o in resp.json())
        return resp.json()

        # Save face vals into csv
        # df = None
        # if df is None:
        #     df = pd.DataFrame(flatObj) # , columns = ['faceId', 'faceAttributes', 'faceRectangle']
        # else:
        #     df = df.append(pd.DataFrame(flatObj), ignore_index=True)

        # df.to_csv('data.csv')