import json
import requests
import common
import config
import typing

def fetch_target_subtitles_quest(page_url: str, target_va: list[str]) -> dict[str, list[str]]:
    data: dict[str, typing.Any] = common.request_retry_wrapper(lambda: requests.get(page_url).json())
    subtitles: dict[str, list[str]] = {}
    if data.get('data'):
        for i in data['data'].get('storyList', []):
            for j_k in i.get('story', {}):
                j = i['story'][j_k]
                for k in j.get('taskData', []):
                    for l_k in k.get('items', {}):
                        l = k['items'][l_k]
                        if l['role'] in target_va:
                            if subtitles.get(l['role']) is None:
                                subtitles[l['role']] = []
                            t = ' '.join(m['text'] for m in l['text'])
                            subtitles[l['role']].append(t)
                            common.log(f"Fetched subtitle for {l['role']}: {t}")
        return subtitles                    
    else:
        common.log(f"Failed to fetch subtitles for {page_url}.")
        return {}
    

def fetch_target_quests(page_url: str, target_va: list[str]) -> dict[str, list[str]]:
    data: dict[str, typing.Any] = common.request_retry_wrapper(lambda: requests.get('https://gi.yatta.moe/api/v2/EN/quest').json())
    if data.get('data'):
        ids = []
        for i_k in data['data'].get('items', {}):
            i = data['data']['items'][i_k]
            if i['chapterNum'].startswith('page_url'):
                ids += i['id']
        
        return [f'yatta:https://gi.yatta.moe/api/v2/EN/quest/{i}' for i in ids]
    else:
        common.log(f"Failed to fetch quests for {page_url}.")
        return []