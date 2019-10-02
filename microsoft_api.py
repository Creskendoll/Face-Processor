import requests
import cv2
import pandas as pd
import os
from flatten_json import flatten
import json
from io import BytesIO
import sys

subscription_key = "" 
emotion_recognition_url = "https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect"
frames_path = sys.argv[1]

if not os.path.isdir(frames_path):
    print("ERR: Folder not found: %s" % frames_path)
    quit()

if subscription_key == "":
    print("ERR: subscription_key can not be empty! Check line 10 in index.py")
    quit()

df = None

for root, dirs, files in os.walk(frames_path):
    for name in files:
        print("Processing:", name)
        frame = open(os.path.join(root, name), "rb")
        
        header = {'Ocp-Apim-Subscription-Key': subscription_key, "Content-Type": "application/octet-stream"  }

        params = {
            'returnFaceId': 'true',
            'returnFaceLandmarks': 'false',
            'returnFaceAttributes': 'age,gender,smile,emotion'
        }

        resp = requests.post(emotion_recognition_url, params=params, headers=header, data=frame)

        if not resp.status_code == 200:
            print("ERR: Bad request:", resp.text)
            quit()

        # Flatten the responce 
        flatObj = (flatten(o) for o in resp.json())

        if df is None:
            df = pd.DataFrame(flatObj) # , columns = ['faceId', 'faceAttributes', 'faceRectangle']
        else:
            df = df.append(pd.DataFrame(flatObj), ignore_index=True)

        # for obj in resp.json():
        #     flatObj = (flatten(o) for o in obj)
        #     if df is None:
        #         df = pd.DataFrame(flatObj) # , columns = ['faceId', 'faceAttributes', 'faceRectangle']
        #     else:
        #         df.append(pd.DataFrame(flatObj), ignore_index=True)

df.to_csv('data.csv')