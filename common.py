import pathlib
import json
import config
import hashlib
import multiprocessing
import time
import datetime
import typing
import asyncio
from multiprocessing import cpu_count
import threading


def panic(what: str):
    raise RuntimeError(what)


def log(msg: typing.Any):
    print(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S][AIDub] "), msg)


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


processes : list[threading.Thread] = []

cache = {}

missing = []


def cleanup_processes():
    global processes
    while processes:
        p = processes.pop()
        p.join()


def get_free_worker(func, args_list, max_workers=32) -> int:
    """
    Get a free worker to run the given function in parallel. If there are no free workers, create a new one.

    :param func: The function to run in parallel.
    :param args_list: A list of arguments to pass to the function.
    :param max_workers: The maximum number of workers to run in parallel. If None, use the number of CPUs.
    :return: The index of the worker to use.
    """
    global processes
    p = threading.Thread(target=func, args=args_list)
    for i in range(len(processes)):
        if not processes[i].is_alive():
            processes[i] = p
            return i

    # no free worker found, create a new one
    if len(processes) < max_workers:
        processes.append(p)
        return len(processes) - 1
    else:
        i = 0
        processes[i].join()
        processes[i] = p
        return i


def run_in_parallel(func, args_list, max_workers=32):
    """
    Run the given function in parallel using the maximum number of workers. If there are no free workers, create a new one.
    :param func: The function to run in parallel.
    :param args_list: A list of arguments to pass to the function.
    :param max_workers: The maximum number of workers to run in parallel. If None, use the number of CPUs.
    :return: None.
    """
    global processes

    if max_workers is None:
        max_workers = cpu_count()

    worker = get_free_worker(func, args_list, max_workers)
    processes[worker].start()


def check_if_audio_exceeds_10s(audio_path: str) -> bool:
    import soundfile as sf
    audio, sr = sf.read(audio_path)
    log(f"Audio length: {len(audio) / sr}s")
    return (len(audio) / sr) > 9 or (len(audio) / sr) < 4


def request_retry_wrapper(fetcher: typing.Callable, max_retries: int = 128):
    for _ in range(max_retries):
        try:
            req = fetcher()
            return req
        except Exception as e:
            import random
            log(f"Failed to fetch data due to {e}, retrying...")
            time.sleep(1 / random.randint(1, 5))
            continue
        
    log(f"Failed to fetch data after {max_retries} retries.")
    return None


def dataset_overview():
    dataset_manifest = json.loads(pathlib.Path(config.dataset_manifest_file_dest).read_text())
    log(f"File: {config.dataset_manifest_file_dest}")
    log(f"Contain characers: {dataset_manifest.keys()}")
    log(f"Total number of samples: {sum(len(dataset_manifest[char]) for char in dataset_manifest)}")
    for i in dataset_manifest:
        log(f"{i}: {len(dataset_manifest[i])} samples")
        
        
def cached_data(key: str, data_resolver: typing.Callable):
    global cache

    cache_item = cache.get(key, None)
    if cache_item is not None:
        return cache_item
    
    cache[key] = data_resolver()
    return cache[key]


def get_available_model_path() -> dict[str, tuple[str, str]]:
    import re
    gpt_path = pathlib.Path('thirdparty/GPTSoViTs/GPT_weights_v2')
    sovits_path = pathlib.Path('thirdparty/GPTSoViTs/SoVITS_weights_v2')
    model_paths = {}
    for ckpt, pth in zip(gpt_path.iterdir(), sovits_path.iterdir()):
        try:
            if ckpt is not None:
                # trunc from the first char including `(` `)` to first - or _
                model_name = re.match(r"^(.*?)([-_])", ckpt.stem).group(1)
                # convert the first char to uppercase
                model_name = model_name[0:1].upper() + model_name[1:]
                val = model_paths.get(model_name, ["", ""])
                val[0] = ckpt
                model_paths[model_name] = val
            if pth is not None:
                # trunc from the first char to first _ or -
                model_name = re.match(r"^(.*?)([-_])", pth.stem).group(1)
                # convert the first char to uppercase
                model_name = model_name[0:1].upper() + model_name[1:]
                val = model_paths.get(model_name, ["", ""])
                val[1] = pth
                model_paths[model_name] = val
        except:
            pass
    
    for i in model_paths:
        model_paths[i] = (str(model_paths[i][0]), str(model_paths[i][1]))
    # exclude those not in muted_chars
    global missing
    muted_chars = get_muted_chars()
    
    for i in list(model_paths.keys()):
        if i not in muted_chars:
            log(f"Model {i} is not in muted_chars, please add it to the config.")
            del model_paths[i]
            missing.append(i)
    
    missing = list(set(missing))
        
    return model_paths


def get_default_model_path() -> tuple[str, str]:
    return ("thirdparty/GPTSoViTs/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "thirdparty/GPTSoViTs/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth")


def get_muted_chars() -> list[str]:
    try:
        s1 = [i for i in json.loads(pathlib.Path(config.dataset_manifest_file_dest).read_text()).keys()]
        s2 = missing
        return [i for i in set(s1 + s2)]
    except FileNotFoundError:
        return []


def extract_character_name(encoded_char_name: str) -> tuple[str, str]:
    """
    Extract the character name and language from the encoded name.

    Args:
        encoded_char_name (str): The encoded character name.

    Returns:
        tuple[str, str]: The character name and language.
    """
    if '(' not in encoded_char_name:
        return encoded_char_name, 'en'
    
    # find the last occurrence of (xxx) and recognize it as lang
    lang = encoded_char_name.split('(')[-1].split(')')[0]
    # remove the lang from the encoded name
    encoded_char_name = encoded_char_name.replace(f'({lang})', '')
    return encoded_char_name, lang


def encode_character_name(char_name: str, lang: str) -> str:
    """
    Encode the character name and language into a single string.

    Args:
        char_name (str): The character name.
        lang (str): The language.

    Returns:
        str: The encoded character name.
    """
    return f'{char_name}({lang})'


language_mapping = {
    'Chinese': 'zh',
    'English(US)': 'en',
    'Japanese': 'jp'
}