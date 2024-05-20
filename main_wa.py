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

emotion_factor = {
    'good': 1,
    'neu': 0.98,
    'bad': 0.3
}

text_classification = {
    'admiration': 'good',
    'amusement': 'good',
    'anger':'bad', 
    'annoyance':'bad', 
    'approval':'good', 
    'caring':'good', 
    'confusion':'bad', 
    'curiosity':'neu', 
    'desire':'neu', 
    'disappointment':'bad', 
    'disapproval':'neu', 
    'disgust':'bad', 
    'embarrassment':'bad', 
    'excitement':'good', 
    'fear':'bad', 
    'gratitude':'neu', 
    'grief':'bad', 
    'joy':'good', 
    'love':'neu', 
    'nervousness':'bad', 
    'optimism':'neu', 
    'pride':'neu', 
    'realization':'neu', 
    'relief':'neu', 
    'remorse':'bad', 
    'sadness':'bad', 
    'surprise':'neu',
    'neutral':'neu'
}

audio_classification = {
    'neu':'neu',
    'ang':'bad',
    'hap':'good',
    'sad':'bad'
}

video_classification = {
    "Angry":'bad',
    "Disgust":'bad',
    "Fear":'bad', 
    "Happy":'good',
    "Neutral":'neu', 
    "Sad":'bad',
    "Surprise":'neu'
}


def run_camera(camera, video_emotions):
    while True:
        frame, pred = Cam.get_pred_frame()
        video_emotions.append(pred)
        imshow("Camera", frame)
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
            print("Error: ", text_resp)
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

    temp = 0
    count = 0
    for i in video_emotions:
        if i != "":
            temp += emotion_factor[video_classification[i]]
            count += 1
    
    temp/=count

    video_emotions.clear()

    if audio_emotions['Statement'] is None:
        continue

    res = (35*emotion_factor[text_classification[audio_emotions['Text']['Emotion']]]+45*emotion_factor[audio_classification[audio_emotions['Audio']['Emotion']]]+20*temp)

    print(video_emotions)
    print(audio_emotions)
    print(f'RES = {res}')