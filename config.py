muted_characters = [
    # "Kinich",
    # "Kachina",
    # "Nahida",
    # "Tighnari",
    # "Candace",
    # "Dehya",
    # "Collei",
    # "Dori",
    # "Kaveh",
    # "Alhaitham",
    # "Nilou",
    # "Faruzan"
    # "Kinich",
    # "Yoimiya",
    # "Yoimiya(zh)",
    # "Xilonen"
]


ignored_characters = [
    "???",
    "Crowd",
    "Everyone",
    "(TravelerTravelerThe player's chosen name for the Traveler)"
]


muted_language = "en"
enable_multilingual = True # experimental feature, when enabled, the character name will recognize things in () as languages

models_path = {
    "Kinich": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/kinich-e15.ckpt",  # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/kinich_e10_s360.pth"  # SoVITS
    ),
    "Kachina": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/kachina-e15.ckpt",  # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/kachina_e8_s408.pth"  # SoVITS
    ),
    "Candace": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Candace-e15.ckpt",  # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Candace_e8_s160.pth"  # SoVITS
    ),
    "Nahida": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Nahida-e10.ckpt",  # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Nahida_e8_s432.pth"  # SoVITS
    ),
    "Tighnari": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Tighnari-e15.ckpt",  # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Tighnari_e8_s240.pth"  # SoVITS
    ),
    "Dehya": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Dehya-e15.ckpt",  # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Dehya_e8_s312.pth"  # SoVITS
    ),
    "Collei": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Collei-e15.ckpt",  # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Collei_e8_s192.pth"  # SoVITS
    ),
    "Dori": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Dori-e15.ckpt",  # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Dori_e8_s184.pth"  # SoVITS
    ),
    "Kaveh": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Kaveh-e15.ckpt",  # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Kaveh_e8_s128.pth"  # SoVITS
    ),
    "Alhaitham": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Alhaitham-e15.ckpt",  # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Alhaitham_e8_s616.pth"  # SoVITS
    ),
    "Nilou": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Nilou-e15.ckpt",  # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Nilou_e8_s160.pth"  # SoVITS
    ),
    "default": (
        "thirdparty/GPTSoViTs/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "thirdparty/GPTSoViTs/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    )
}

# sources to fetch voice
sources_to_fetch_voice = [
    "https://genshin-impact.fandom.com/wiki/Black_Stone_Under_a_White_Stone",
    "https://genshin-impact.fandom.com/wiki/Flowers_Resplendent_on_the_Sun-Scorched_Sojourn",
    "https://genshin-impact.fandom.com/wiki/Kinich%27s_Deal",
    "https://genshin-impact.fandom.com/wiki/Lingering_Warmth",
    "https://genshin-impact.fandom.com/wiki/Homecoming",
    "https://genshin-impact.fandom.com/wiki/The_Unanswerable_Problems",
    "https://genshin-impact.fandom.com/wiki/Lupus_Aureus_Chapter",
    "https://genshin-impact.fandom.com/wiki/Sands_of_Solitude",
    "https://genshin-impact.fandom.com/wiki/Oathkeeper",
    "https://genshin-impact.fandom.com/wiki/King_Deshret_and_the_Three_Magi",
    "https://genshin-impact.fandom.com/wiki/Dreams,_Emptiness,_Deception",
    "https://genshin-impact.fandom.com/wiki/The_Morn_a_Thousand_Roses_Brings",
    "https://genshin-impact.fandom.com/wiki/Through_Mists_of_Smoke_and_Forests_Dark",
    "https://genshin-impact.fandom.com/wiki/Oathkeeper",
    "https://genshin-impact.fandom.com/wiki/Floral_Debt,_Blood_Due",
    "https://genshin-impact.fandom.com/wiki/The_Unanswerable_Problems",
    "https://genshin-impact.fandom.com/wiki/Sands_of_Solitude",
    "https://genshin-impact.fandom.com/wiki/The_Illusions_of_the_Mob",
    "custom:genshin_huggingface:foobar",
    "custom:hsr_huggingface:foobar"
]


# text to dub
# add "quest:" before the quest page which contains the dialogue part
source_text_to_dub = [
    "quest:https://genshin-impact.fandom.com/wiki/For_the_Same_Land",
    "https://honkai-star-rail.fandom.com/wiki/A_New_Venture_on_the_Eighth_Dawn"
]

necessary_replacements = {
    "TravelerTravelerThe player's chosen name for the Traveler": "Traveler",
    "He'sHe'sText for male Traveler/She'sShe'sText for female Traveler": "He",
    "himhimText for male Traveler/herherText for female Traveler": "him",
}

# save destination
save_dest_root = "saves"
save_dest_for_downloaded_voice = f"{save_dest_root}/downloaded_voices"
dataset_manifest_file_dest = f"{save_dest_root}/dataset_manifest.json"
dub_manifest_dest = f"{save_dest_root}/dub_manifest.json"
dub_result_dest = f"{save_dest_root}/dub_result"
dub_result_manifest_dest = f"{save_dest_root}/dub_result_manifest.json"
sentiment_analysis_dest = f"{save_dest_root}/sentiment_analysis.json"
screenshot_dest = f"{save_dest_root}/screenshot.png"

# gpt sovits configuration
logs_dest = f"{save_dest_root}/logs"
gpt_model_path = "thirdparty/GPTSoViTs/GPT_weights_v2"
sovits_model_path = "thirdparty/GPTSoViTs/SoVITS_weights_v2"

# Python 3.10 adaption
import pathlib
import sys
if sys.version_info.major == 3 and sys.version_info.minor <= 10:
    print("Python 3.10 detected, applying adaptions...")
    # patch read_text
    pathlib.Path._real_read_text = pathlib.Path.read_text
    pathlib.Path.read_text = lambda self, encoding='utf-8', errors=None: self._real_read_text(encoding=encoding, errors=errors)
    # patch write_text
    pathlib.Path._real_write_text = pathlib.Path.write_text
    pathlib.Path.write_text = lambda self, data, encoding='utf-8', errors=None: self._real_write_text(data=data, encoding=encoding, errors=errors)