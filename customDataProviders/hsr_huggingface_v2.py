import common
import datasets
import soundfile as sf

language = ['Chinese', 'English', 'Japanese', 'Korean']
language_rep = ['zh', 'en', 'jp', 'ko']
language_mapping = {
    'Chinese(PRC)': 'zh',
    'English': 'en',
    'English(US)': 'en',
    'Japanese': 'jp',
    'Korean': 'ko'
}

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

dataset = common.request_retry_wrapper(lambda: datasets.load_dataset("simon3000/starrail-voice", split="train", streaming=True))



def fetch_vo_urls(url: str, target_va: list[str]) -> dict[str, tuple[str, str]]:
    """
    Fetches the urls of the voices of the given target_va from the given url.
    Returns a dictionary with the voice names as keys and a tuple of the voice url and the voice audio format as values.
    """
    data = dataset.filter(lambda voice: common.encode_character_name(voice['speaker'], language_mapping[voice['language'] if voice['language'] != "" else 'English']) in target_va or
                        (voice['speaker'] in target_va and voice['language'] == 'English'))
    
    result = {}
    
    for i in data:
        name = common.encode_character_name(i['speaker'], language_mapping[i['language']])
        
        if result.get(name, None) is None: 
            result[name] = []
        if len(result[name]) > 300:
            continue
        if i['language'] == 'Chinese':
            for k, v in replace_dict_cn.items():
                i['transcription'] = i['transcription'].replace(k, v)
        elif i['language'] == 'English(US)':
            for k, v in replace_dict_en.items():
                i['transcription'] = i['transcription'].replace(k, v)
        if '{' in i['transcription']:
            common.log(f"Unreplaced tags in {i['speaker']}: {i['transcription']}, skipping...")
            continue
        
        import io, pathlib
        pth = pathlib.Path('./hsr_huggingface_cache') / f"cache_{common.md5(i['audio']['path'])}.mp3"
        pth.parent.mkdir(exist_ok=True, parents=True)
        sf.write(str(pth), i['audio']['array'], i['audio']['sampling_rate'],format='mp3')
        
        result[name].append((i['transcription'], {
            'type': 'mp3',
            'path': str(pth)
        }))
        
    return result