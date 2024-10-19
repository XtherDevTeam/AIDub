import time
import fuzzywuzzy.fuzz
import mss.screenshot
import mss.tools
import config
import json
import pathlib
import pyscreenshot
import pynput
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import common
import fuzzywuzzy
import nltk

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="en")

collections = json.loads(pathlib.Path(config.dub_result_manifest_dest).read_text())

def get_bottom_half_screenshot():
    pyscreenshot.grab().save(config.screenshot_dest)
    cv2_img = cv2.imread(config.screenshot_dest)
    height, width, channels = cv2_img.shape
    bottom_half = cv2_img[height * 2 // 3:, :]
    cv2.imwrite(config.screenshot_dest, bottom_half)
    
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
        for item in page:
             extracted_texts.append(item[1][0])

    return "".join(extracted_texts)
    
def extract_text():
    ocr_result = ocr.ocr(config.screenshot_dest, cls=True)
    return extract_text_from_ocr_result(ocr_result)

played = {}

def check_similarity(text1, ori_text):
    hits, misses = (0, 0)
    
    text1 = text1.lower().replace(" ", "").replace("\n", "")
    words = nltk.word_tokenize(ori_text.lower())
    print(words, text1)
    for i in words:
        if i in text1:
            hits += 1
        else:
            misses += 1
    
    return hits / (hits + misses)


def play(dest: str):
    from pydub import AudioSegment
    from pydub.playback import play

    #convert audio to datasegment
    sound = AudioSegment.from_file(dest, "aac") 
    play(sound)  #play sound


def match_once():
    global played
    get_bottom_half_screenshot()
    text = extract_text()
    for char in collections:
        if char in text:
            # there is a character name, means potential character subtitle, do full check
            for subtitle_label in collections[char]:
                ori_text = collections[char][subtitle_label]['text']
                similarities = [(check_similarity(text, ori_text), collections[char][subtitle_label]['dest']) for i in collections[char][subtitle_label]]
                similarities.sort(reverse=True, key=lambda x: x[0])
                if similarities[0][0] > 0.5 or played.get(ori_text, True):
                    common.log(f"Found subtitle {subtitle_label} for character {char}, similarity {similarities[0][0]}, playing {similarities[0][1]}")
                    play(similarities[0][1])
                    played[ori_text] = False
                    return True
    common.log(f"No subtitle found for text, skipping")
    return False
            

def daemon_wrapper():
    while True:
        match_once()
        time.sleep(0.5)
        

def run_dnp():
    common.log("Starting DNP")
    daemon_wrapper()