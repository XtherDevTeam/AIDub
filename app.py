import argparse
from multiprocessing import cpu_count

import common
import config
import detect_and_play
import dub
import fandom
import finetune
import voice_fetch
from config import sources_to_fetch_voice, muted_characters, dataset_manifest_file_dest, source_text_to_dub, \
    dub_manifest_dest


def do_voice_collection():
    """
    collections: dict[str, list[tuple[str, str]]] = {}
    
    for i in sources_to_fetch_voice:
        
        quests = fandom.fetch_quest_entries(i)
        

        for quest in quests:
            collection = fandom.fetch_target_vo_from_quest_page(quest, muted_characters)

            collections = fandom.merge_voice_collections([collections, collection])

    # download voices
    # voice_fetch.reduce_collection(collections)
    voice_fetch.fetch_collection(collections)

    with open(dataset_manifest_file_dest, 'w') as f:
        f.write(voice_fetch.serialize_collection(collections))
    """

    voice_fetch.generate_text_list()
    

def do_subtitle_collection():
    collections: dict[str, list[str]] = {}
    for i in source_text_to_dub:
        quests = fandom.fetch_quest_entries(i)

        for quest in quests:
            collection = fandom.fetch_target_subtitles(quest, muted_characters)

            collections = fandom.merge_subtitle_collections([collections, collection])

    with open(dub_manifest_dest, 'w') as f:
        f.write(dub.serialize_collection(collections))

def do_finetune():
    finetune.start()

if __name__ == '__main__':
    # use argparse to parse command line arguments

    parser = argparse.ArgumentParser()
    parser.description = 'AI Dubbing for Genshin Impact Natlan Region due to the EN VA Strike'
    parser.add_argument('--voice', action='store_true', help='Fetch voices for dataset')
    parser.add_argument('--subtitle', action='store_true', help='Fetch subtitles for dubbing')
    parser.add_argument('--finetune', action='store_true', help='Finetune the model')
    parser.add_argument('--inference-server', action='store_true', help='Start fish-speech inference server for dubbing')
    parser.add_argument('--dub-all', action='store_true', help='Dub all the subtitles in the manifest file')
    parser.add_argument('--detect-and-play', action='store_true', help='Run the wrapper script to detect and play the dubbed audio')
    args = parser.parse_args()

    if args.voice:
        do_voice_collection()
    elif args.subtitle:
        do_subtitle_collection()
    elif args.finetune:
        do_finetune()
    elif args.dub_all:
        dub.dub_all()
    elif args.inference_server:
        dub.run_gpt_sovits_api_server()
    elif args.detect_and_play:
        detect_and_play.run_dnp()
    else:
        parser.print_help()