from Mic.mic import MicAudio
from Camera.camera import CameraFeed
from cv2 import imshow, waitKey, destroyAllWindows
import threading
from sklearn.neural_network import MLPClassifier
import pickle

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

def extract_video(video_emotions):
    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]
    count = [video_emotions.count(emotion) for emotion in EMOTIONS_LIST]
    mode = max(count)
    if mode == 0:
        mode = 1
    return [num // mode for num in count]


def extract_audio(audio_emotions):
    EMOTIONS_LIST = ['neu', 'ang', 'hap', 'sad']
    audio_input = [0] * len(EMOTIONS_LIST)
    emotion = audio_emotions['Audio']['Emotion']
    audio_input[EMOTIONS_LIST.index(emotion)] = 1
    return audio_input


def extract_text(audio_emotions):
    EMOTIONS_LIST = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    text_input = [0] * len(EMOTIONS_LIST)
    emotion = audio_emotions['Text']['Emotion']
    text_input[EMOTIONS_LIST.index(emotion)] = 1
    return text_input


def aggr_emotions(audio_emotions, video_emotions, clf):
    video_input = extract_video(video_emotions)
    audio_input = extract_audio(audio_emotions)
    text_input = extract_text(audio_emotions)

    input = [*video_input, *audio_input, *text_input]
    prediction = clf.predict(input)
    print(prediction)


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

clf = MLPClassifier(hidden_layer_sizes=(150,150,150), max_iter=3000, verbose=1, random_state=21, tol=0.000000001)
clf = pickle.load(open('finalized_model.sav', 'rb'))

t1 = threading.Thread(target=run_camera, args=(Cam, video_emotions))
t1.start()

while True:
    t2 = threading.Thread(target=run_mic, args=(Mic, 12, audio_emotions))
    t2.start()
    t2.join()

    t3 = threading.Thread(target=aggr_emotions, args=(audio_emotions, video_emotions, clf))
    t3.start()
