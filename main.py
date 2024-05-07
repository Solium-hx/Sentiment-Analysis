from Mic.mic import MicAudio
from Camera.camera import CameraFeed
from cv2 import imshow, waitKey, destroyAllWindows
import threading

audio_emotions = {
    'Statement': None,
    'Text': {
        'Emotion': None,
        'Score': None,
    },
    'Audio': {
        'Emotion': None,
        'Score': None,
    }
}

video_emotions = []

def aggr_emotions(audio_emotions, video_emotions):
    print("Video: ", video_emotions)
    print("Statement: ", audio_emotions['Statement'])
    print("Text: ", audio_emotions["Text"])
    print("Audio: ", audio_emotions["Audio"])


def run_camera(camera, video_emotions):
    while True:
        frame, pred = Cam.get_pred_frame()
        video_emotions.append(pred)
        imshow("Camera", frame)
        if pred != "":
            print(pred)
        if waitKey(1) == 27:
            break
    destroyAllWindows()

def run_mic(mic, duration, audio_emotions):
    def text_analysis(audio, audio_emotions):
        success, text_resp = mic.run_text_analysis(resp)
        if success:
            audio_emotions['Statement'] = text_resp[0]
            audio_emotions['Text'] = {
                'Emotion': text_resp[1][0]['label'],
                'Score': text_resp[1][0]['score']
            }
        else:
            print("Error: ", resp)
            audio_emotions = {
                'Statement': None,
                'Text': {
                    'Emotion': None,
                    'Score': None,
                }
            }
    
    def audio_analysis(audio, audio_emotions):
        out_prob, score, index, text_lab = mic.run_audio_analysis(resp)
        audio_emotions['Audio'] = {
            'Emotion': text_lab[0],
            'Score': score.item()
        }

    success, resp = mic.get_audio(duration=duration)
    if success:
        t1 = threading.Thread(target=text_analysis, args=(resp, audio_emotions, ))
        t2 = threading.Thread(target=audio_analysis, args=(resp, audio_emotions, ))

        t1.start()
        t2.start()

        t1.join()
        t2.join()
    else:
        print("Error: ", resp)
        audio_emotions = {
            'Statement': None,
            'Text': {
                'Emotion': None,
                'Score': None,
            }
        }
        audio_emotions['Audio'] = {
            'Emotion': None,
            'Score': None,
        }
            

Cam = CameraFeed()
Mic = MicAudio()

t1 = threading.Thread(target=run_camera, args=(Cam, video_emotions))
t1.start()

while True:
    t2 = threading.Thread(target=run_mic, args=(Mic, 12, audio_emotions))
    t2.start()
    t2.join()

    t3 = threading.Thread(target=aggr_emotions, args=(audio_emotions, video_emotions, ))
    t3.start()
