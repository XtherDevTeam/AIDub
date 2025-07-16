import pathlib
import json
import config
import common
from transformers import pipeline
import torch
import random
import thirdparty.speechbrainModel

# Load the BERT-Emotions-Classifier
use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
classifier = None
audio_classifier = None


analysis_file = {}


def load_bert_classifier():
    global classifier
    if classifier is None:
        classifier = pipeline("text-classification", model="michellejieli/emotion_text_classifier", device=use_device)


def load_audio_classifier():
    global audio_classifier
    if audio_classifier is None:
        audio_classifier = pipeline("audio-classification", model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3", device=use_device)

def audio_classifier_mapping(results_audio: list[dict], results_text: list[dict]) -> tuple[str, float]:
    #  happy angry neutral disgust fearful surprised calm to joy anger neutral disgust fear surprise and neutral.
    mapping = {
        "neutral": ["neutral"],
        "angry": ["anger"],
        "disgust": ["disgust"],
        "fearful": ["sadness"],
        "happy": ["joy"],
        "surprised": ["surprise"],
        "calm": ["neutral"],
        "sad": ['sadness']
    }
    
    if results_audio[0]['label'] == 'fearful':
        return results_text[0]['label'], results_text[0]['score']
    else:
        return mapping[results_audio[0]['label']][0], results_audio[0]['score']
        

def classify_emotion(text) -> str:
    load_bert_classifier()
    results = classifier(text)
    return results[0]['label']


def classify_audio_emotion(audio_path: str) -> str:
    load_audio_classifier()
    results = audio_classifier(audio_path)
    return results[0]['label']

def load_analysis_file():
    if not pathlib.Path(config.sentiment_analysis_dest).exists():
        initialize_analysis_file()
    global analysis_file
    analysis_file = json.loads(pathlib.Path(
        config.sentiment_analysis_dest).read_text())
    return analysis_file


def initialize_analysis_file():
    file_struct = {}
    dataset = json.loads(pathlib.Path(
        config.dataset_manifest_file_dest).read_text())
    for char in dataset:
        common.log(f"Initializing analysis file for {char}")
        file_struct[char] = {
            # anger	disgust	fear	joy	neutral	sadness	surprise
            "anger": [],
            "disgust": [],
            "fear": [],
            "joy": [],
            "neutral": [],
            "sadness": [],
            "surprise": []
        }
    pathlib.Path(config.sentiment_analysis_dest).write_text(
        json.dumps(file_struct))
    global analysis_file
    analysis_file = file_struct

def save_analysis_file():
    pathlib.Path(config.sentiment_analysis_dest).write_text(
        json.dumps(analysis_file))


def do_batch_classification(char):
    load_audio_classifier()
    load_bert_classifier()
    dataset = json.loads(pathlib.Path(
        config.dataset_manifest_file_dest).read_text())
    if char not in dataset:
        common.log(f"Character {char} not found in dataset")
        return
    for hash_id in dataset[char]:
        try:
            x = dataset[char][hash_id]
            text = x
            audio_emotion = audio_classifier(text["dest"])
            text_emotion = classifier(text["text"])
            label, score = audio_classifier_mapping(audio_emotion, text_emotion)
            
            analysis_file[char][label].append({
                "text": text['text'],
                "hash_id": hash_id,
                "score": score,
                "dest": x["dest"]
            })
        except RuntimeError as e:
            common.log(f"Error processing {hash_id} for {char}: {e}, ignoring")
        
    common.log(f"Classification of {char} complete")
    

def choose_a_voice_by_text(char: str, text: str, randomed: bool = True) -> dict[str, str] | None:
    """
    Chooses a most representative voice for the given text based on the sentiment analysis file.

    Args:
        char (str): The character to dub
        text (str): The text to analyze

    Returns:
        dict[str, str] | None: A dictionary containing the hash_id, text, score, and destination of the most representative voice for the given text. If no suitable voice is found, returns None.
    """
    load_bert_classifier()
    result = classifier(text)[0]
    label = result['label']
    
    # rank the analysises by score
    ranked_analysis = sorted(analysis_file[char][label], key=lambda x: x['score'], reverse=True)
    def resolver(ranked_analysis):
        result = []
        for analysis in ranked_analysis:
            try:
                if not common.check_if_audio_exceeds_10s(analysis['dest']):
                    result.append(analysis)
            except Exception as e:
                common.log(f"Error processing {analysis['hash_id']}: {e}, ignoring")
        return result
    
    ranked_analysis = common.cached_data(f"ranked_analysis_{char}_{label}", lambda: resolver(ranked_analysis))
    return random.choice(ranked_analysis) if len(ranked_analysis) > 0 else None

    
def do_classification():
    """
    Do the sentiment analysis on all the characters in the dataset.
    """
    initialize_analysis_file()
    dataset = json.loads(pathlib.Path(
        config.dataset_manifest_file_dest).read_text())
    common.log(f"Starting classification of {len(dataset)} characters")
    for char in dataset:
        do_batch_classification(char)
    common.log(f"Classification of all characters complete")
    save_analysis_file()