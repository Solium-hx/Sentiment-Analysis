import cv2
from .model import ERModel
import numpy as np

facec = cv2.CascadeClassifier('Camera/haarcascade_frontalface_default.xml')
video_model = ERModel("Camera/model.json", "Camera/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

def get_pred_frame(path):
    video = cv2.VideoCapture(path)
    video_emotions = []
    while video.isOpened():
        ret, frame = video.read()
        if ret == True:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rect = facec.detectMultiScale(gray_img)
            pred = ""
            for (x, y, w, h) in face_rect:
                fc = gray_img[y:y+h, x:x+w]
                roi = cv2.resize(fc, (48, 48))
                pred = video_model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            video_emotions.append(pred)
        else:
            return video_emotions
