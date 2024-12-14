# fandom.com operations related
import av
import time
import urllib
import config
import bs4
import requests
import urllib.parse

import common


def fetch_quest_entries_from_chapter_page(page_url: str) -> list[str]:
    """
    This function takes a chapter page URL and returns a list of quest entries present in the chapter.

    Params:
    page_url (str): URL of the chapter page.

    Returns:
    list[str]: List of quest entries present in the chapter.
    """
    collection = []
    req = common.request_retry_wrapper(lambda: requests.get(page_url))
    req.encoding = "utf-8"
    document = bs4.BeautifulSoup(req.text, "html.parser")
    # find all <li> tags with ::marker
    # ol : list[bs4.element.Tag]
    li : bs4.element.Tag
    for ol in document.find_all("ol"):
        for li in ol.find_all("li"):
            label = li.find("a")
            if label:
                real_url = urllib.parse.urljoin(page_url, label.get("href"))
                collection.append(real_url)
    return collection

def fetch_quest_entries_from_tribe_quest_page(page_url: str) -> list[str]:
    """
    This function takes a tribe quest page URL and returns a list of quest entries present in the tribe quest.

    Params:
    page_url (str): URL of the tribe quest page.

    Returns:
    list[str]: List of quest entries present in the tribe quest.
    """
    collection = []
    req = common.request_retry_wrapper(lambda: requests.get(page_url))
    
    req.encoding = "utf-8"
    document = bs4.BeautifulSoup(req.text, "html.parser")
    # find all <ul> tags
    li : bs4.element.Tag
    # start from div class="mw-parser-output"
    div = document.find('div', {'class':'mw-parser-output'})
    if div is None:
        common.log(f"unexpected fandom page: {page_url}")
        return collection

    for ul in div.find_all("ul"):
        for li in ul.find_all("li"):
            label = li.find("a")
            if label and li.text.strip().startswith("Act"):
                real_url = urllib.parse.urljoin(page_url, label.get("href"))
                collection.append(real_url)

    common.log(collection)
    return collection

def fetch_quest_entries(input: str):
    common.log(f"Fetching quest entries from {input}")
    if input.startswith('tribe:'):
        # tribe quest, get url after :
        url = input[6:]
        common.log(f"Fetching quest entries from tribe quest page: {url}")
        return fetch_quest_entries_from_tribe_quest_page(url)
    else:
        # chapter, get url
        return fetch_quest_entries_from_chapter_page(input)


def fetch_target_vo_from_quest_page(page_url: str, target_va: list[str]) -> dict[str, list[tuple[str, str]]]:
    """
    This function takes a quest page URL and returns the target VO of the quest.

    Params:
    page_url (str): URL of the quest page.

    Returns:
    dict[str, list[tuple[str, str]]]: Dictionary containing the VO of each target VA of the quest.
    """

    collection = {}

    req: requests.Response = common.request_retry_wrapper(lambda: requests.get(page_url))
    
    req.encoding = "utf-8"
    document = bs4.BeautifulSoup(req.text, "html.parser")
    dialogueParts = document.find_all('div', {'class': 'dialogue'})
    if dialogueParts is None:
        common.log(f"unexpected dialogue part: {page_url}")
        return collection
    else:
        common.log(f"Found dialogue part for {page_url}")

    for dialoguePart in dialogueParts:
        ddLabels = dialoguePart.find_all(name='dd')
        for i in ddLabels:
            bLabel = i.find('b')
            if bLabel is None:
                # useless dialogue, skip
                continue

            char = bLabel.text[0:-1]

            if char in target_va:
                if i.find('span') is None:
                    common.log(f"No vocal file found for character: {char} in text: {i.text}")
                    continue
                src = i.find('span').find('a').attrs['href']
                common.log(f"Found character {char} vocal file: {src}")
                # remove other texts
                text = i.get_text()
                text = text[text.find(f'{char}: ') + len(f'{char}: '):]
                for i in config.necessary_replacements:
                    common.log(f"Replacing {i} with {config.necessary_replacements[i]} in text: {text}")
                    text = text.replace(i, config.necessary_replacements[i])

                if collection.get(char) is None:
                    collection[char] = []
                collection[char].append((text, src))
                
    # workaround for hsr wiki
    # find all span with Play
    playableVoiceLines = dialoguePart.find_all("span", {"title": "Play"})
    for i in playableVoiceLines:
        i.parent #dd label
        char = i.parent.find('b').text[0:-1]
        text = i.parent.get_text()
        text = text[text.find(f'{char}: ') + len(f'{char}: '):]
        
        for i in config.necessary_replacements:
            common.log(f"Replacing {i} with {config.necessary_replacements[i]} in text: {text}")
            text = text.replace(i, config.necessary_replacements[i])
            
        aLabel = i.find('a')
        if aLabel is None:
            common.log(f"No vocal file found for character: {char} in text: {i.text}")
            continue
        src = aLabel.attrs['href']
        if char in target_va:
            common.log(f"Found character {char} subtitle: {text}")
            if collection.get(char) is None:
                collection[char] = []
            collection[char].append((text, src))

    return collection


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


def fetch_target_subtitles(page_url: str, target_va: list[str]) -> dict[str, list[str]]:
    """
    This function takes a chapter page URL and returns the target VO of the chapter.

    Params:
    page_url (str): URL of the chapter page.

    Returns:
    dict[str, list[tuple[str, str]]]: Dictionary containing the VO of each target VA of the chapter.
    """
    collection = {}

    req = common.request_retry_wrapper(lambda: requests.get(page_url))
    req.encoding = "utf-8"
    document = bs4.BeautifulSoup(req.text, "html.parser")
    dialogueParts = document.find_all('div', {'class': 'dialogue'})
    
    if dialogueParts is None:
        common.log(f"unexpected dialogue part: {page_url}")
        return collection

    for dialoguePart in dialogueParts:
        ddLabels = dialoguePart.find_all(name='dd')
        for i in ddLabels:
            bLabel = i.find('b')
            if bLabel is None:
                # useless dialogue, skip
                continue

            char = bLabel.text[0:-1]

            if char in target_va:
                if i.find('span') is None:
                    common.log(f"No subtitle found for character: {char} in text: {i.text}, trying fallback method")
                    if i.text.strip().startswith(f'{char}: '):
                        text = i.text.strip()[len(f'{char}: '):]
                        common.log(f"Found character {char} subtitle: {text}")
                        if collection.get(char) is None:
                            collection[char] = []
                        collection[char].append(text)
                    else:
                        common.log(f"No subtitle found for character: {char} in text: {i.text}")
                    continue

                # get text after `char:`
                text = i.text
                text = text[text.find(f'{char}: ') + len(f'{char}: '):]
                common.log(f"Found character {char} subtitle: {text}")
                if collection.get(char) is None:
                    collection[char] = []
                collection[char].append(text)

        # workaround for hsr wiki
        # find all span with no-audio
        noVoiceLabels = dialoguePart.find_all("span", {"title": "Invalid or missing audio file"})
        for i in noVoiceLabels:
            i.parent #dd label
            char = i.parent.find('b').text[0:-1]
            text = i.parent.get_text()
            text = text[text.find(f'{char}: ') + len(f'{char}: '):]
            if char in target_va:
                common.log(f"Found character {char} subtitle: {text}")
                if collection.get(char) is None:
                    collection[char] = []
                collection[char].append(text)
    return collection


def merge_subtitle_collections(collections: list[dict[str, list[str]]]):
    """
    This function takes a list of collections and merge them into a single collection.

    Params:
    collections (list[dict[str, list[str]]]): List of collections.

    Returns:
    dict[str, list[str]]: Merged collection.
    """
    merged = {}
    for collection in collections:
        for char, sub_list in collection.items():
            if merged.get(char) is None:
                merged[char] = []
            merged[char].extend(sub_list)
    return merged


def find_potentially_missing_voice_over_chars(page_url: str, target_va: list[str]) -> list[str]:
    """
    

    Params:
    page_url (str): URL of the chapter page.

    Returns:
    
    """
    import threading
    result = []
    threads = []

    req = common.request_retry_wrapper(lambda: requests.get(page_url))
    req.encoding = "utf-8"
    document = bs4.BeautifulSoup(req.text, "html.parser")
    dialogueParts = document.find_all('div', {'class': 'dialogue'})
    
    if dialogueParts is None:
        common.log(f"unexpected dialogue part: {page_url}")
        return []

    for dialoguePart in dialogueParts:
        ddLabels = dialoguePart.find_all(name='dd')
        for i in ddLabels:
            bLabel = i.find('b')
            if bLabel is None:
                # useless dialogue, skip
                continue

            char = bLabel.text[0:-1]
            # print(f"Checking character {char}", i.text.strip())
            if char not in target_va and f"{char}:" in i.text.strip() and char not in config.ignored_characters and '&' not in char:
                if i.find('span') is None:
                    common.log(f"Possible missing character {char} voiceline: {i.text}")
                    result.append(char)
                else:
                    # check if voice is usable
                    x = i.find('span')
                    if x.find('a') is None:
                        common.log(f"Possible missing character {char} voiceline: {i.text}")
                        result.append(char)
                    else:
                        def func():
                            try:
                                src = x.find('a').attrs['href']
                                audio = av.open(src)
                                if audio.duration < 1:
                                    common.log(f"Possible missing character {char} voiceline: {i.text}")
                                audio.close()
                                result.append(char)
                            except:
                                common.log(f"Failed to load {src} for character {char}")
                                result.append(char)
                        threads.append(threading.Thread(target=func))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    return [i for i in set(result)]