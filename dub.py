import emotion
import GPTSoVits
import common
import config
import os
import json
import pathlib
import requests

# models_path = config.models_path
# muted_characters = config.muted_characters

def serialize_collection(collection: dict[str, list[str]]):
    return json.dumps(collection)

def make_dirs():
    os.makedirs(config.dub_result_dest, exist_ok=True)

def pick_random(collection: dict[str, list[str]], char: str) -> tuple[str, str]:
    prompt_voice_lbls = [i for i in collection[char].keys()]
    # pick 1 random key using random lib
    import random
    prompt_voice_lbl = random.sample(prompt_voice_lbls, 1)[0]
    prompt_voice = collection[char][prompt_voice_lbl]
    while common.check_if_audio_exceeds_10s(prompt_voice['dest']):
        common.log(f"Audio exceeds 10s, picking another voice for {char}")
        prompt_voice_lbl, prompt_voice = (random.sample(prompt_voice_lbls, 1)[0], collection[char][prompt_voice_lbl])
        
    return prompt_voice_lbl, prompt_voice


def pick_by_emotion(collection: dict[str, list[str]], char: str, text: str) -> tuple[str, str]:
    d = emotion.choose_a_voice_by_text(char, text)
    if d is None:
        common.log(f"No voice found for {char} with text {text}")
        return None, None
    
    return d['hash_id'], collection[char][d['hash_id']]


def generate_prompt_from_voice(char: str, text: str) -> tuple[str, str]:
    """
    Generates a prompt from a voice file and returns the label and text of the prompt.

    Args:
        char (str): The character to generate the prompt for.
        
    Returns:
        tuple[str, str]: The text and destination of the prompt.
    """
    # select prompt voice
    collection = json.loads(pathlib.Path(config.dataset_manifest_file_dest).read_text())
    
    _, voice = pick_by_emotion(collection, char, text)
    if voice is None:
        common.log(f"No voice found for {char} with text {text}, fallback to random voice")
        _, voice = pick_random(collection, char)
    
    return voice['text'], voice['dest']
    
    
def dub_one(text: str, char: str, raw_response: bool = False) -> tuple[str, str] | requests.Response:
    label = common.md5(text)
    dest_path = os.path.join(config.dub_result_dest, f"{label}.aac")
    if os.path.exists(dest_path) and char not in ['Dan Heng']:
        common.log(f"Dub already exists for {text} for {char}")
        return label, dest_path
    pTexts, pDests = generate_prompt_from_voice(char, text)
    
    if raw_response:
        resp = GPTSoVitsAPI.tts(pDests, pTexts, text, 'auto', 'auto', True)
    
        if resp.status_code != 200:
            common.panic(f"Error generating dub for {text} for {char}: {resp.text}")
            
        return resp
    else:
        resp = GPTSoVitsAPI.tts(pDests, pTexts, text, 'auto', 'auto')
    
        if resp.status_code != 200:
            common.panic(f"Error generating dub for {text} for {char}: {resp.text}")
        
        pathlib.Path(dest_path).write_bytes(resp.content)
        return label, dest_path
    
    
    
def run_gpt_sovits_api_server():
    os.system(f"python thirdparty/GPTSoVITs/api_v2.py -a 127.0.0.1 -p 2372 -c templates/tts_infer.yaml")
    
GPTSoVitsAPI: GPTSoVits.GPTSoVitsAPI = None
    
def setup_gpt_sovits_client(ckpt: str, pth: str):
    global GPTSoVitsAPI
    GPTSoVitsAPI = GPTSoVits.GPTSoVitsAPI('http://127.0.0.1:2372', True, ckpt, pth)
    
    
    
def get_tts_models(char: str,):
    # ckpt, pth
    print(config.models_path)
    return config.models_path.get(char, common.get_default_model_path()) if char != "Albedo" else common.get_default_model_path()
    
    
    
def dub_all(use_middleware_logic):
    if use_middleware_logic:
        config.models_path = common.get_available_model_path()
        config.models_path = common.get_available_model_path()
        config.muted_characters = common.get_muted_chars()
        config.muted_characters = common.get_muted_chars()
        
    make_dirs()
    emotion.load_analysis_file()
    collection = json.loads(pathlib.Path(config.dub_manifest_dest).read_text())
    dub_result_manifest = {}
    for char in collection:
        ckpt, pth = get_tts_models(char)
        setup_gpt_sovits_client(ckpt, pth)
        for text in collection[char]:
            common.log(f"Dubbing {text} for {char}")
            label, dest_path = dub_one(text, char)
            if dub_result_manifest.get(char) is None:
                dub_result_manifest[char] = {}
            
            dub_result_manifest[char][label] = {
                "dest": dest_path,
                "text": text,
                "label": label
            }

    pathlib.Path(config.dub_result_manifest_dest).write_text(serialize_collection(dub_result_manifest))