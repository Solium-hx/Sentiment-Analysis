from Mic.mic import MicAudio
from Camera.camera import CameraFeed
from cv2 import imshow, waitKey, destroyAllWindows
import threading

audio_emotions = {}
video_emotions = []

def aggr_emotions(audio_emotions, video_emotions):
    print(video_emotions)
    print(audio_emotions["Statement"])
    print(audio_emotions["Text"])
    print(audio_emotions["Audio"])


def run_camera(camera):
    while True:
        frame, pred = Cam.get_pred_frame()
        video_emotions.append(pred)
        imshow("Camera", frame)
        if waitKey(1) == 27:
            break
    destroyAllWindows()

def run_mic(mic, duration):
    def text_analysis(audio):
        success, text_resp = mic.run_text_analysis(resp)
        if success:
            audio_emotions['Statement'] = text_resp[0]
            audio_emotions['Text'] = {
                'Emotion': text_resp[1][0]['label'],
                'Score': text_resp[1][0]['score']
            }
        else:
            print("Error: ", resp)
    
    def audio_analysis(audio):
        out_prob, score, index, text_lab = mic.run_audio_analysis(resp)
        audio_emotions['Audio'] = {
            'Emotion': text_lab[0],
            'Score': score.item()
        }

    success, resp = mic.get_audio(duration=duration)
    if success:
        t1 = threading.Thread(target=text_analysis, args=(resp, ))
        t2 = threading.Thread(target=audio_analysis, args=(resp, ))

        t1.start()
        t2.start()

        t1.join()
        t2.join()
    else:
        print("Error: ", resp)
            

Cam = CameraFeed()
Mic = MicAudio()

t1 = threading.Thread(target=run_camera, args=(Cam, ))
t1.start()

while True:
    t2 = threading.Thread(target=run_mic, args=(Mic, 12, ))
    t2.start()
    t2.join()

    t3 = threading.Thread(target=aggr_emotions, args=(audio_emotions, video_emotions))
    t3.start()
