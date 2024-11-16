import requests
import urllib.parse
import tqdm
import json
import threading
import time
import common

# lanuage to use
language = ['Chinese']

src = "https://datasets-server.huggingface.co/rows?dataset=simon3000%2Fgenshin-voice&config=default&split=train"
chunk_count = 32
thread_pool = []
result = {}

replace_dict_cn = {
    '{NICKNAME}': '旅行者',
    '{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_HE|INFO_MALE_PRONOUN_SHE]}': '他',
    '{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_SHE|INFO_MALE_PRONOUN_HE]}': '她',
    '{M#她}{F#他}': '她'
}

replace_dict_en = {
    '{NICKNAME}': 'Traveler',
    '{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_HE|INFO_MALE_PRONOUN_SHE]}': 'he',
    '{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_SHE|INFO_MALE_PRONOUN_HE]}': 'she',
    '{M#she}{F#he}': 'she',
    '{M#her}{F#his}': 'her',
    '{M#his}{F#her}': 'his',
    '{M#sister}{F#brother}': 'brother',
    '{M#sister}{F#sister}':'sister',
}


def get_chunk_size(chunk_pos):
    total = 413429
    split = total // chunk_count
    # return [chunk_pos*chunk_count, min(total, (chunk_pos+1)*chunk_count)]
    return [chunk_pos*split, min(total, (chunk_pos+1)*split)]


def get_data(offset, length):
    global src
    url = f"{src}&offset={offset}&length={length}"
    response = requests.get(url)
    data = json.loads(response.text)
    return data


def get_specific_speaker_language(data, speakers, languages):
    """Fetch the data and get the voice collections of the given speakers and language."""
    r = {}
    for _ in data['rows']:
        i = _['row']
        if i['speaker'] in speakers and i['language'] in languages:
            if r.get(i['speaker']) is None:
                r[i['speaker']] = []
            
            # remove annoying tags
            for k, v in replace_dict_cn.items():
                i['transcription'] = i['transcription'].replace(k, v)
            for k, v in replace_dict_en.items():
                i['transcription'] = i['transcription'].replace(k, v)
            
            # text, src
            r[i['speaker']].append((i['transcription'], i['audio'][0]['src']))
    return r


def merge_voice_collections(collections: list[dict[str, list[tuple[str, str]]]]) -> dict[str, list[tuple[str, str]]]:
    """
    This function takes a list of collections and merge them into a single collection.

    Params:
    collections (list[dict[str, list[tuple[str, str]]]]): List of collections.

    Returns:
    dict[str, list[tuple[str, str]]]: Merged collection.
    """
    merged = {}
    for collection in collections:
        for char, vo_list in collection.items():
            if merged.get(char) is None:
                merged[char] = []
            merged[char].extend(vo_list)
    return merged


def traverse_api(speakers, language, chunk):
    chunk_pos, chunk_size = get_chunk_size(chunk)
    print(chunk_pos, chunk_size)
    r = {}
    offset = 0
    length = 100
    # while offset < total:
    # use range to rewrite the loop
    for offset in tqdm.tqdm(range(chunk_pos, chunk_size, 100), position=chunk):
        while True:
            try:
                raw_data = get_data(offset, min(chunk_size-offset, length))
                collection = get_specific_speaker_language(
                    raw_data, speakers, language)
                break
            except Exception as e:
                print(raw_data)

            time.sleep(1)
        r = merge_voice_collections([r, collection])
    
    result = merge_voice_collections([result, r])


def start_multi_threads(speakers, language):
    for i in range(chunk_count):
        th = threading.Thread(target=traverse_api, args=(speakers, language, i))
        th.start()
        thread_pool.append(th)
    for th in thread_pool:
        th.join()


def fetch_vo_urls(url: str, target_va: list[str]) -> dict[str, tuple[str, str]]:
    """
    Fetches the urls of the voices of the given target_va from the given url.
    Returns a dictionary with the voice names as keys and a tuple of the voice url and the voice audio format as values.
    """
    start_multi_threads(target_va, language)
    return result