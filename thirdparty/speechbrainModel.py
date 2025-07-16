from speechbrain.inference.interfaces import foreign_class

class SpeeechBrainPipeline():
    def __init__(self):
        self.classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
        
    def __call__(self, audio_file):
        out_prob, score, index, text_lab = self.classifier.classify_file(audio_file)
        return [{
            'label': text_lab,
            'score': score
        }]
