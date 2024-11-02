muted_characters = ["Kinich", "Kachina", "Nahida", "Tighnari", "Candace", "Dehya", "Collei"]
muted_language = "en"

models_path = {
    "Kinich": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/kinich-e15.ckpt", # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/kinich_e10_s360.pth" # SoVITS
    ),
    "Kachina": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/kachina-e15.ckpt", # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/kachina_e8_s408.pth" # SoVITS
    ),
    "Candace": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Candace-e15.ckpt", # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Candace_e8_s160.pth" # SoVITS
    ),
    "Nahida": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Nahida-e10.ckpt", # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Nahida_e8_s432.pth" # SoVITS
    ),
    "Tighnari": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Tighnari-e15.ckpt", # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Tighnari_e8_s240.pth" # SoVITS
    ),
    "Dehya": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Dehya-e15.ckpt", # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Dehya_e8_s312.pth" # SoVITS
    ),
    "Collei": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/Collei-e15.ckpt", # GPT
        "thirdparty/GPTSoViTs/SoVITS_weights_v2/Collei_e8_s192.pth" # SoVITS
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
    "https://genshin-impact.fandom.com/wiki/The_Unanswerable_Problems"
]


# text to dub
# add "quest:" before the quest page which contains the dialogue part
source_text_to_dub = [
    "https://genshin-impact.fandom.com/wiki/Beyond_the_Smoke_and_Mirrors",
    "https://genshin-impact.fandom.com/wiki/The_Rainbow_Destined_to_Burn",
    "quest:https://genshin-impact.fandom.com/wiki/Give_Her_Surprises",
    "quest:https://genshin-impact.fandom.com/wiki/Give_Her_Sweetness",
    "quest:https://genshin-impact.fandom.com/wiki/Give_Her_Memories"
]

# save destination

save_dest_root = "saves"
save_dest_for_downloaded_voice = f"{save_dest_root}/downloaded_voices"
dataset_manifest_file_dest = f"{save_dest_root}/dataset_manifest.json"
dub_manifest_dest = f"{save_dest_root}/dub_manifest.json"
dub_result_dest = f"{save_dest_root}/dub_result"
dub_result_manifest_dest = f"{save_dest_root}/dub_result_manifest.json"
sentiment_analysis_dest = f"{save_dest_root}/sentiment_analysis.json"
screenshot_dest = f"{save_dest_root}/screenshot.png"

# fish speech configuration
fish_speech_module_path = "thirdparty/fish"