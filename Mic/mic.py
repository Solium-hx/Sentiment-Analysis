import speech_recognition as sr
from speechbrain.inference.interfaces import foreign_class
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

audio_classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

text_tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
text_model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
text_classifier = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

class MicAudio(object):
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def get_audio(self, duration):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.record(source, duration=duration)
                return True, audio

        except sr.RequestError as e:
            return False, e
        
        except sr.UnknownValueError as e:
            return False, e
        
    def run_text_analysis(self, audio):
        try:
            text = self.recognizer.recognize_google(audio)
            emotion = text_classifier(text)
            return True, [text, emotion]

        except Exception as e:
            return False, e
        
    def run_audio_analysis(self, audio):
        with open("temp_audio.wav", "wb") as f:
            f.write(audio.get_wav_data())
        return audio_classifier.classify_file("temp_audio.wav")
        