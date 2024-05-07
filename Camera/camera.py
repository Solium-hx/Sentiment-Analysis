import cv2
from .model import ERModel
import numpy as np

facec = cv2.CascadeClassifier('Camera/haarcascade_frontalface_default.xml')
video_model = ERModel("Camera/model.json", "Camera/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX


class CameraFeed(object):
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            print("Error: Camera connection not established")
        
    def __del__(self):
        self.cam.release()

    def get_pred_frame(self):
        while self.cam.isOpened():
            ret, frame = self.cam.read()
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rect = facec.detectMultiScale(gray_img)
            pred = ""
            for (x, y, w, h) in face_rect:
                fc = gray_img[y:y+h, x:x+w]
                roi = cv2.resize(fc, (48, 48))
                pred = video_model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                
            return frame, pred
