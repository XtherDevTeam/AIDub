import bettercam
import pydub
import re
import pydub.playback
import pydub.silence
import pylcs
import time
import fuzzywuzzy.fuzz
import mss.screenshot
import mss.tools
import config
import json
import pathlib
import pynput
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import common
import fuzzywuzzy
import nltk

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="en", ocr_version="PP-OCRv4", )

collections = json.loads(pathlib.Path(config.dub_result_manifest_dest).read_text())

def trim_leading_silence(sound: pydub.AudioSegment) -> pydub.AudioSegment:
    """
    从音频段的开头修剪静音。

    Args:
        sound: AudioSegment 对象

    Returns:
        修剪后的 AudioSegment 对象，或原始对象（如果未检测到静音）。
    """
    silence_ranges = pydub.silence.detect_silence(sound, min_silence_len=100, silence_thresh=sound.dBFS - 16)  # 可调整参数

    if silence_ranges:
        first_nonsilent_start = silence_ranges[0][1]
        trimmed_sound = sound[first_nonsilent_start:]
        return trimmed_sound
    else:
        return sound
    
def extract_text_from_ocr_result(ocr_result):
    """
    Extracts text from an OCR result in the given format.

    Args:
        ocr_result: A list of lists, where each inner list contains bounding box 
                   coordinates and the recognized text.

    Returns:
        A list of strings, where each string is the recognized text.
    """

    extracted_texts = []
    for page in ocr_result:  # Iterate through pages (if multiple)
        if page is None:  # Skip empty pages
            continue
        
        for item in page:
             extracted_texts.append(item[1][0])

    return "".join(extracted_texts)
    
def extract_text(image: np.ndarray):
    ocr_result = ocr.ocr(image, cls=True)
    return extract_text_from_ocr_result(ocr_result)

played = {}

def check_similarity(text1, ori_text, char):
    # replace all non-alphanumeric characters with space
    text1 = re.sub(r'[^a-zA-Z0-9]+','', text1)
    # cut from `{charName}: ` to `uidxxx`
    rightpos = text1.lower().find(f"uid")
    if rightpos == -1:
        rightpos = len(text1)
    text1 = text1.lower()[len(char):rightpos]
    ori_text = re.sub(r'[^a-zA-Z0-9]+','', ori_text)
    ori_text = ori_text.lower()
    # get the longest common substring length
    min_len = min(len(text1), len(ori_text))
    min_len = min_len if min_len > 0 else 1
    pylcs_result = pylcs.lcs_string_length(text1, ori_text) / min_len
    similarity = fuzzywuzzy.fuzz.ratio(text1, ori_text) / 100
    if similarity > 0.5 and pylcs_result > 0.5:
        return (pylcs_result + similarity, len(ori_text), True, pylcs_result, text1, ori_text)
    
    return (0, len(ori_text), False, 0, text1, ori_text) # no similarity, return 0
    


def play(dest: str):
    #convert audio to datasegment
    sound = pydub.AudioSegment.from_file(dest, "aac")
    sound = trim_leading_silence(sound)  # remove leading silence
    pydub.playback.play(sound)  #play sound


def check_if_played(text):
    global played
    common.log(f"{text} in {played.keys()}: {text in played.keys()}")
    if text in played.keys():
        return True
    return False


def update_played(text):
    global played
    played[text] = True


def match_once(image: np.ndarray):
    global played
    text = extract_text(image)
    for char in collections:
        if char in text:
            
            # there is a character name, means potential character subtitle, do full check
            similarities = [(check_similarity(text, collections[char][i]['text'], char), collections[char][i]['dest'], collections[char][i]['text']) for i in collections[char]]
            
            similarities.sort(reverse=True, key=lambda x: x[0][0])
            ori_text = similarities[0][2]
            
            common.log(similarities[0:5])
            # if (similarities[0][0][0] > 0.5 and similarities[0][0][1] > 30) or (similarities[0][0][0] > 0.7 and similarities[0][0][1] <= 30) or played.get(ori_text, True):
            # check if matched
            if similarities[0][0][2] and not check_if_played(ori_text):
                common.log(f"Found subtitle {ori_text} for character {char}, similarity {similarities[0][0][0]}, playing {similarities[0][1]}")
                play(similarities[0][1])
                update_played(ori_text)
                return True
    common.log(f"No subtitle found for text, skipping")
    return False
            

def daemon_wrapper():
    camera = bettercam.create()
    # get screen size
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        width = monitor["width"]
        height = monitor["height"]
    # calculate the bottm 1/3 of the screen
    bottom_height = int(height * 2/3)
    camera.start(region=(0, height-bottom_height, width, height))
    timer = time.time()
    while True:
        match_once(camera.get_latest_frame())
        time.sleep(0.2)
        
    camera.stop()
        
        

def run_dnp():
    common.log("Starting DNP")
    daemon_wrapper()