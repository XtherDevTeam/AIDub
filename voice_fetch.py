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
import threading

def make_dirs():
    os.makedirs(config.save_dest_for_downloaded_voice, exist_ok=True)

def download_task_wrapper(char: str, text: str, url: str):
    def download_task():
        out = pathlib.Path(config.save_dest_for_downloaded_voice) / char / f"{common.md5(text)}.mp3"
        if out.exists() and not out.read_bytes().startswith(b"<?xml"):
            
            return
        
        if type(url) == dict:
            os.rename(url['path'], str(out))
        else:
            r = requests.get(url, allow_redirects=True, stream=False, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'})
            if r.status_code == 403:
                common.log(f"Failed to download {url} for {char} with status code {r.status_code}")
                common.log(f"Skipping {char} for this time")
                return
            if r.status_code != 200:
                common.log(f"Failed to download {url} for {char} with status code {r.status_code}")
                raise Exception("Failed to download")
            with open(out, 'wb+') as file:
                total = int(r.headers.get('content-length'))
                for i in clint.textui.progress.bar(r.iter_content(chunk_size=2391975), expected_size=(total / 1024) + 1):
                    if i:
                        file.write(i)
    return common.request_retry_wrapper(lambda: download_task())


def dispatch_download_task_with_fallback(char: str, text: str, furl: str, fallback_url: str, cache_path: str, index: int):
    def download_task():
        out = pathlib.Path(config.save_dest_for_downloaded_voice) / char / f"{common.md5(text)}.mp3"
        url = furl
        if out.exists() and not out.read_bytes().startswith(b"<?xml"):
            return
        
        import time
        pth = pathlib.Path(cache_path)
        print(pth, pth.stat().st_mtime, time.time(), pth.stat().st_mtime > time.time() - 86400)
        if pth.exists() and pth.stat().st_mtime > time.time() - 86400: # check whether the files modified today
            # use cache to refresh url
            try:
                content = pth.read_text()
                url = json.loads(content)['rows'][index]['row']['audio'][0]['src']
                print(json.loads(content)['rows'][index], url)
                assert url != ''
            except:
                # otherwise, use fallback url
                print(f"Failed to load cache for {char}, using fallback url: {fallback_url}")
                content = common.request_retry_wrapper(lambda: requests.get(fallback_url)).content
                content = content.decode('utf-8')
                pth.write_text(content)
                url = json.loads(content)['rows'][index]['row']['audio'][0]['src']
                print(json.loads(content)['rows'][index], url)
                assert url != ''
        else:
            # otherwise, use fallback url
            content = common.request_retry_wrapper(lambda: requests.get(fallback_url)).content
            content = content.decode('utf-8')
            pth.write_text(content)
            url = json.loads(content)['rows'][index]['row']['audio'][0]['src']
            print(json.loads(content)['rows'][index], url)
            assert url != ''
            
        if type(url) == dict:
            os.rename(url['path'], str(out))
        else:
            r = requests.get(url, allow_redirects=True, stream=False, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'})
            if r.status_code == 403:
                common.log(f"Failed to download {url} for {char} with status code {r.status_code}")
                common.log(f"Skipping {char} for this time")
                return
            if r.status_code != 200:
                common.log(f"Failed to download {url} for {char} with status code {r.status_code}")
                raise Exception("Failed to download")
            with open(out, 'wb+') as file:
                print(out, 'path')
                total = int(r.headers.get('content-length'))
                for i in clint.textui.progress.bar(r.iter_content(chunk_size=2391975), expected_size=(total / 1024) + 1):
                    if i:
                        file.write(i)
    return common.request_retry_wrapper(lambda: download_task())


def dispatch_download_task(char: str, text: str, url: str):
    x = lambda: download_task_wrapper(char, text, url)
    # common.run_in_parallel(, ())
    x()


def fetch_collection(collection: dict[str, list[tuple[str, str, str, str]]]):
    make_dirs()
    for char in collection:
        ths: list[threading.Thread] = []
        os.makedirs(pathlib.Path(config.save_dest_for_downloaded_voice) / char, exist_ok=True)
        # seperate 8 list with same counts of elements
        def split_list(lst, n):
            common.log(f"{lst}")
            return [lst[i:i+n] for i in range(0, len(lst), n)]
        x = split_list(collection[char], 48)
        
        def dispatcher(lst):
            for voice in lst:
                # judge tuple size
                if len(voice) == 5:
                    text, url, fallback_url, cache_path, index = voice
                    dispatch_download_task_with_fallback(char, text, url, fallback_url, cache_path, index)
                else:
                    text, url = voice
                    dispatch_download_task(char, text, url)
        for i in x:
            ths.append(threading.Thread(target=dispatcher, args=(i,)))
        for i in ths:
            i.start()
        for i in ths:
            i.join()



def serialize_collection(collection: dict[str, list[tuple[str, str, str, str]]]) -> str:
    result = {}
    for char in collection:
        for voice in collection[char]:
            label = common.md5(voice[0])
            if len(voice) == 5:
                text, url, fallback_url, cache_path, index = voice
            else:
                text, url = voice
            if result.get(char) is None:
                result[char] = {}
            result[char][label] = {"text": text, "url": url, "dest": os.path.join(config.save_dest_for_downloaded_voice, char, f"{label}.mp3")}

    # persist unicodes
    # merge with existing data
    existing_data = {}
    if pathlib.Path(config.dataset_manifest_file_dest).exists():
        existing_data = json.loads(pathlib.Path(config.dataset_manifest_file_dest).read_text())
    for char in list(existing_data.keys()) + list(result.keys()):
        if char in existing_data:
            if char in result:
                result[char] = {**existing_data[char], **result[char]}
            else:
                result[char] = existing_data[char]

    return json.dumps(result, ensure_ascii=False)

def reduce_collection(collection: dict[str, list[tuple[str, str, str, str]]]) -> dict[str, list[tuple[str, str, str, str]]]:
    not_enough_data = []
    former = json.loads(pathlib.Path(config.dataset_manifest_file_dest).read_text())
    for char in collection:
        # fetch 10 elements for each character
        save_keys = [i for i in collection[char]][0:300]
        collection[char] = save_keys
        if len(collection[char]) < 10:
            not_enough_data.append(char)
    for i in not_enough_data:
        if former.get(i) is not None:
            common.log(f"Using existing data for {i}, ")
            collection[i] = former[i]
        else:
            common.log(f"Not enough data for {i}, removing from collection")
            del collection[i]
        
    return collection
    
def reduce_collection_by_size(collection: dict[str, list[tuple[str, str, str, str]]], min_size: int = 22210) -> dict[str, list[tuple[str, str, str, str]]]:
    # for each character in collection, check each audio's size, perserve only those with size > min_size
    reduced_collection = {}
    for char in collection:
        reduced_collection[char] = []
        for voice in collection[char]:
            path = os.path.join(config.save_dest_for_downloaded_voice, char, f"{common.md5(voice[0])}.mp3")
            if os.path.exists(pathlib.Path(path)):
                size = os.path.getsize(pathlib.Path(path))
            else:
                size = 0
            if size > min_size:
                reduced_collection[char].append(voice)
    return reduced_collection
    
def generate_text_list(colab_project_prefix: pathlib.Path = pathlib.Path(config.save_dest_for_downloaded_voice)) -> list[str]:
    collection = json.loads(pathlib.Path(config.dataset_manifest_file_dest).read_text())
    for char in collection:
        generated = ""
        for i in collection[char]:
            text, dest = collection[char][i]['text'], collection[char][i]['dest']
            # vocal_path|speaker_name|language|text
            text = text.replace('\n', '')
            generated += f"{pathlib.Path(dest).name}|{char}|{common.extract_character_name(char)[1]}|{text}" + "\n"
        
            # with open(, "w+") as f:
            #     f.write(generated)
            pth: pathlib.Path = pathlib.Path(config.save_dest_for_downloaded_voice) / char / f"{char}.list"
            pth.write_text(generated)