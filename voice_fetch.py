import asyncio
import os
import pathlib
import subprocess

import clint
import requests

import common
import config
import fandom
import json

def make_dirs():
    os.makedirs(config.save_dest_for_downloaded_voice, exist_ok=True)

def download_task_wrapper(char: str, text: str, url: str):
    def download_task():
        out = pathlib.Path(config.save_dest_for_downloaded_voice) / char / f"{common.md5(text)}.mp3"
        if out.exists():
            return
        # fake our ua
        r = requests.get(url, allow_redirects=True, stream=True, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'})
        with open(out, 'wb+') as file:
            total = int(r.headers.get('content-length'))
            for i in clint.textui.progress.bar(r.iter_content(chunk_size=2391975), expected_size=(total / 1024) + 1):
                if i:
                    file.write(i)
    return common.request_retry_wrapper(lambda: download_task())

def dispatch_download_task(char: str, text: str, url: str):
    common.run_in_parallel(download_task_wrapper, (char, text, url))


def fetch_collection(collection: dict[str, list[tuple[str, str]]]):
    make_dirs()
    for char in collection:
        os.makedirs(pathlib.Path(config.save_dest_for_downloaded_voice) / char, exist_ok=True)

        for voice in collection[char]:
            text, url = voice
            dispatch_download_task(char, text, url)

def serialize_collection(collection: dict[str, list[tuple[str, str]]]) -> str:
    result = {}
    for char in collection:
        for voice in collection[char]:
            label = common.md5(voice[0])
            text, url = voice
            if result.get(char) is None:
                result[char] = {}
            result[char][label] = {"text": text, "url": url, "dest": os.path.join(config.save_dest_for_downloaded_voice, char, f"{label}.mp3")}

    # persist unicodes
    return json.dumps(result, ensure_ascii=False)

def reduce_collection(collection: dict[str, list[tuple[str, str]]]) -> dict[str, list[tuple[str, str]]]:
    for char in collection:
        # fetch 10 elements for each character
        save_keys = [i for i in collection[char]][0:]
        collection[char] = save_keys
        
    return collection
    
def generate_text_list(colab_project_prefix: pathlib.Path = pathlib.Path(config.save_dest_for_downloaded_voice)) -> list[str]:
    collection = json.loads(pathlib.Path(config.dataset_manifest_file_dest).read_text())
    for char in collection:
        generated = ""
        for i in collection[char]:
            text, dest = collection[char][i]['text'], collection[char][i]['dest']
            # vocal_path|speaker_name|language|text
            text = text.replace('\n', '')
            generated += f"{pathlib.Path(dest).name}|{char}|{config.muted_language}|{text}" + "\n"
        
            with open(pathlib.Path(config.save_dest_for_downloaded_voice) / char / f"{char}.list", "w+") as f:
                f.write(generated)