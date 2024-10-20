import GPTSoVits
import common
import config
import os
import json
import pathlib

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
    return prompt_voice_lbl, prompt_voice

def check_if_audio_exceeds_10s(audio_path: str) -> bool:
    import soundfile as sf
    audio, sr = sf.read(audio_path)
    common.log(f"Audio length: {len(audio) / sr}s")
    return (len(audio) / sr) > 9 or (len(audio) / sr) < 4

def generate_prompt_from_voice(char: str) -> tuple[str, str]:
    """
    Generates a prompt from a voice file and returns the label and text of the prompt.

    Args:
        char (str): The character to generate the prompt for.
        
    Returns:
        tuple[str, str]: The text and destination of the prompt.
    """
    # select prompt voice
    collection = json.loads(pathlib.Path(config.dataset_manifest_file_dest).read_text())
    _, voice = pick_random(collection, char)
    while check_if_audio_exceeds_10s(voice['dest']):
        common.log(f"Audio exceeds 10s, picking another voice for {char}")
        _, voice = pick_random(collection, char)
    
    return voice['text'], voice['dest']
    
    
def dub_one(text: str, char: str):
    label = common.md5(text)
    dest_path = os.path.join(config.dub_result_dest, f"{label}.aac")
    if os.path.exists(dest_path):
        common.log(f"Dub already exists for {text} for {char}")
        return label, dest_path
    pTexts, pDests = generate_prompt_from_voice(char)
    
    resp = GPTSoVitsAPI.tts(pDests, pTexts, text, 'en', 'en')
    
    if resp.status_code != 200:
        common.panic(f"Error generating dub for {text} for {char}: {resp.text}")
    
    pathlib.Path(dest_path).write_bytes(resp.content)
    
    return label, dest_path
    
    
    
def run_gpt_sovits_api_server():
    os.system(f"python thirdparty/GPTSoViTs/api_v2.py -a 127.0.0.1 -p 9880 -c thirdparty/GPTSoViTs/GPT_SoVITS/configs/tts_infer.yaml")
    
GPTSoVitsAPI: GPTSoVits.GPTSoVitsAPI = None
    
def setup_gpt_sovits_client(ckpt: str, pth: str):
    global GPTSoVitsAPI
    GPTSoVitsAPI = GPTSoVits.GPTSoVitsAPI('http://127.0.0.1:9880', True, ckpt, pth)
    
    
    
def get_tts_models(char: str):
    # ckpt, pth
    return config.models_path[char]
    
    
    
    
def dub_all():
    make_dirs()
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
            
    with open(config.dub_result_manifest_dest, "w") as f:
        f.write(serialize_collection(dub_result_manifest))
