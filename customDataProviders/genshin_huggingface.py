import random
import requests
import urllib.parse
import tqdm
import json
import threading
import time
import common
import pathlib

original_postfix = None # "?Expires=xxxxx&OSSAccessKeyId=xxxxx&Signature=xxxxx"
fixed_speedup_postfix = None # "?Expires=xxxxx&OSSAccessKeyId=xxxxx&Signature=xxxxx"
# lanuage to use
language = ['Chinese', 'English(US)', 'Japanese', 'Korean']
language_rep = ['zh', 'en', 'jp', 'ko']
language_mapping = {
    'Chinese': 'zh',
    'English(US)': 'en',
    'Japanese': 'jp',
    'Korean': 'ko'
}

src = "https://datasets-server.huggingface.co/rows?dataset=simon3000%2Fgenshin-voice&config=default&split=train"
chunk_count = 4096
thread_pool = []
result = {}

replace_dict_cn = {
    '{NICKNAME}': '旅行者',
    '{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_HE|INFO_MALE_PRONOUN_SHE]}': '他',
    '{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_SHE|INFO_MALE_PRONOUN_HE]}': '她',
    '{MATEAVATAR#SEXPRO[INFO_MALE_PRONOUN_HE|INFO_FEMALE_PRONOUN_SHE]}': '她',
    '{MATEAVATAR#SEXPRO[INFO_MALE_PRONOUN_SHE|INFO_FEMALE_PRONOUN_HE]}': '他',
    '{M#她}{F#他}': '她'
}

replace_dict_en = {
    '{NICKNAME}': 'Traveler',
    '{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_HE|INFO_MALE_PRONOUN_SHE]}': 'he',
    '{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_SHE|INFO_MALE_PRONOUN_HE]}': 'she',
    '{MATEAVATAR#SEXPRO[INFO_MALE_PRONOUN_HE|INFO_FEMALE_PRONOUN_SHE]}': 'her',
    '{MATEAVATAR#SEXPRO[INFO_MALE_PRONOUN_SHE|INFO_FEMALE_PRONOUN_HE]}': 'him',
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
    cache = pathlib.Path('./genshin_huggingface_cache') / f'cache_{offset}.json'
    cache.parent.mkdir(exist_ok=True, parents=True)
    if cache.exists(): # cache for 24 hours
        try:
            data = json.loads(cache.read_text())
            if data.get('error') is None:
                return data
        except:
            raise ValueError('Invalid cache file')
    
    url = f"{src}&offset={offset}&length={length}"
    response = requests.get(url, stream=False)
    data = json.loads(response.text)
    cache.write_text(json.dumps(data))
    return data


def result_url_and_cache_path(offset, length):
    return f"{src}&offset={offset}&length={length}", pathlib.Path(f'./genshin_huggingface_cache/cache_{offset}.json')


def get_specific_speaker_language(data, speakers, url, cache_path):
    """Fetch the data and get the voice collections of the given speakers and language."""
    r = {}
    for index, _ in enumerate(data['rows']):
        i = _['row']
        # encode the char name
        speaker_with_lang = common.encode_character_name(i['speaker'], language_mapping.get(i['language'], 'N/A'))
        alternative_if_en = speaker_with_lang[:-4] if speaker_with_lang.endswith('(en)') else 'N/A'
        if speaker_with_lang in speakers or alternative_if_en in speakers:
            final_choice = speaker_with_lang if alternative_if_en == 'N/A' else alternative_if_en
            
            if r.get(final_choice) is None:
                r[final_choice] = []
            
            # remove annoying tags
            if i['language'] == 'Chinese':
                for k, v in replace_dict_cn.items():
                    i['transcription'] = i['transcription'].replace(k, v)
            elif i['language'] == 'English(US)':
                for k, v in replace_dict_en.items():
                    i['transcription'] = i['transcription'].replace(k, v)
                
            if '{' in i['transcription']:
                common.log(f"Unreplaced tags in {i['speaker']}: {i['transcription']}, skipping...")
                continue
            
            # text, src
            r[final_choice].append((i['transcription'], i['audio'][0]['src'], url, str(cache_path), index))
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


def traverse_api(speakers, chunk):
    chunk_pos, chunk_size = get_chunk_size(chunk)
    # print(chunk_pos, chunk_size)
    r = {}
    offset = 0
    length = 100
    # while offset < total:
    # use range to rewrite the loop
    for offset in tqdm.tqdm(range(chunk_pos, chunk_size, 100), position=chunk):
        while True:
            try:
                raw_data = get_data(offset, min(chunk_size-offset, length))
                url, cache_path = result_url_and_cache_path(offset, length)
                collection = get_specific_speaker_language(
                    raw_data, speakers, url, cache_path )
                break
            except ValueError as e:
                # skip
                collection = []
                break
            except Exception as e:
                # print(raw_data)
                time.sleep(1 / random.randint(1, 10))
                
        r = merge_voice_collections([r, collection])
    
    global result
    result = merge_voice_collections([result, r])


def start_multi_threads(speakers):
    for i in range(chunk_count):
        th = threading.Thread(target=traverse_api, args=(speakers, i))
        th.start()
        thread_pool.append(th)
    for th in thread_pool:
        th.join()


def fetch_vo_urls(url: str, target_va: list[str]) -> dict[str, tuple[str, str]]:
    """
    Fetches the urls of the voices of the given target_va from the given url.
    Returns a dictionary with the voice names as keys and a tuple of the voice url and the voice audio format as values.
    """
    start_multi_threads(target_va)
    return result