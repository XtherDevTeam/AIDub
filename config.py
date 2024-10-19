muted_characters = ["Kinich", "Kachina"]
muted_language = "en"

models_path = {
    "Kinich": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/kinich-e15.ckpt", # GPT
        "/mnt/g/WinghongZau/Projects/AIDub/thirdparty/GPTSoViTs/SoVITS_weights_v2/kinich_e10_s360.pth" # SoVITS
    ),
    "Kachina": (
        "thirdparty/GPTSoViTs/GPT_weights_v2/kachina-e15.ckpt", # GPT
        "/mnt/g/WinghongZau/Projects/AIDub/thirdparty/GPTSoViTs/SoVITS_weights_v2/kachina_e8_s408.pth" # SoVITS
    )
}

# sources to fetch voice
# Tribe quests starts with "tribe:"
sources_to_fetch_voice = [
    "https://genshin-impact.fandom.com/wiki/Black_Stone_Under_a_White_Stone",
    "https://genshin-impact.fandom.com/wiki/Flowers_Resplendent_on_the_Sun-Scorched_Sojourn",
    "https://genshin-impact.fandom.com/wiki/Kinich%27s_Deal"
]


# text to dub
source_text_to_dub = [
    "https://genshin-impact.fandom.com/wiki/Beyond_the_Smoke_and_Mirrors",
    "https://genshin-impact.fandom.com/wiki/The_Rainbow_Destined_to_Burn"
]

# save destination

save_dest_root = "saves"
save_dest_for_downloaded_voice = f"{save_dest_root}/downloaded_voices"
dataset_manifest_file_dest = f"{save_dest_root}/dataset_manifest.json"
dub_manifest_dest = f"{save_dest_root}/dub_manifest.json"
dub_result_dest = f"{save_dest_root}/dub_result"
dub_result_manifest_dest = f"{save_dest_root}/dub_result_manifest.json"
screenshot_dest = f"{save_dest_root}/screenshot.png"

# fish speech configuration
fish_speech_module_path = "thirdparty/fish"