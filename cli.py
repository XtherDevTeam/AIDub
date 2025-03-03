import argparse
import os
import pathlib
import typing
import config
import common
import fandom
import voice_fetch
import emotion
import dub
import importlib
import communication

def run_gpt_sovits_server():
    os.system("cd GPT-SoVITs && python api_v2.py")


def before_request():
    config.models_path = common.get_available_model_path()
    config.muted_characters = common.get_muted_chars()
    emotion.load_analysis_file()

def makeResult(ok: bool = True, data: typing.Any = None) -> dict:
    return {
        'status': ok,
        'data': data
    }

def do_voice_collection():
    collections: dict[str, list[tuple[str, str]]] = {}
    for i in config.sources_to_fetch_voice:
        if i.startswith('custom:'):
            providerName = i[7:i.index(':', 7)]
            url = i[i.index(':')+1:]
            provider = importlib.import_module(f'customDataProviders.{providerName}')
            collections = fandom.merge_voice_collections([collections, provider.fetch_vo_urls(url, config.muted_characters)])
        else:
            quests = common.request_retry_wrapper(lambda: fandom.fetch_quest_entries(i))


            for quest in quests:
                collection = fandom.fetch_target_vo_from_quest_page(quest, config.muted_characters)

                collections = fandom.merge_voice_collections([collections, collection])

    # download voices
    voice_fetch.reduce_collection(collections)
    voice_fetch.fetch_collection(collections)
    pathlib.Path(config.dataset_manifest_file_dest).write_text(voice_fetch.serialize_collection(collections))

    voice_fetch.generate_text_list()


def download_dataset_cli(char_names: list[str], sources_to_fetch: list[str]):
    if not char_names or not sources_to_fetch:
        print("Error: --char-names and --sources-to-fetch are required.")
        return False

    config.muted_characters = [i for i in set(char_names + config.muted_characters)]
    config.sources_to_fetch_voice = [i for i in set(sources_to_fetch + config.sources_to_fetch_voice)]
    do_voice_collection()

    print(f"Dataset for {char_names} downloaded successfully.")
    return True


def emotion_classification_cli():
    emotion.do_classification()
    print('Emotion classification completed.')
    return True


def dub_cli(text: str, char_name: str):
    if not text or not char_name:
        print("Error: --text and --char-name are required.")
        return False

    ckpt, pth = dub.get_tts_models(char_name)
    dub.setup_gpt_sovits_client(ckpt, pth)
    r = dub.dub_one(text, char_name, True)

    headers = r.headers
    filename = f'{char_name}.aac'
    headers['Content-Disposition'] = f'attachment; filename={filename}'

    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=10*1024):
            if chunk:
                f.write(chunk)
    print(f"Dubbing completed and saved to {filename}")
    return True


def get_text_cli():
    before_request() # to load config.muted_characters
    success_chars = []
    fail_chars = []
    for char in config.muted_characters:
        try:
            dest = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}")
            list_file = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}/{char}.list")
            communication.preprocess_dataset("1a", char, str(list_file.absolute()), str(dest.absolute()), "0")
            success_chars.append(char)
        except Exception as e:
            print(f"Error while preprocessing dataset for {char}: {e}")
            fail_chars.append(char)

    if fail_chars:
        print(f"Dataset preprocessing (get_text) completed with errors for characters: {fail_chars}. Success for: {success_chars}")
        return False
    else:
        print(f"Dataset preprocessing (get_text) completed successfully for characters: {success_chars}")
        return True

def get_hubert_wav32k_cli():
    before_request() # to load config.muted_characters
    success_chars = []
    fail_chars = []
    for char in config.muted_characters:
        # check if model exists
        if char in config.models_path:
            success_chars.append(char) # skip if model already exists, assuming preprocessed
            continue
        try:
            dest = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}")
            list_file = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}/{char}.list")
            communication.preprocess_dataset("1b", char, str(list_file.absolute()), str(dest.absolute()), "0")
            success_chars.append(char)
        except Exception as e:
            print(f"Error while preprocessing dataset for {char}: {e}")
            fail_chars.append(char)
    if fail_chars:
        print(f"Dataset preprocessing (get_hubert_wav32k) completed with errors for characters: {fail_chars}. Success for: {success_chars}")
        return False
    else:
        print(f"Dataset preprocessing (get_hubert_wav32k) completed successfully for characters: {success_chars}")
        return True


def name_to_semantic_cli():
    before_request() # to load config.muted_characters
    success_chars = []
    fail_chars = []
    for char in config.muted_characters:
        if char in config.models_path:
            success_chars.append(char) # skip if model already exists, assuming preprocessed
            continue
        try:
            dest = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}")
            list_file = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}/{char}.list")
            communication.preprocess_dataset("1c", char, str(list_file.absolute()), str(dest.absolute()), "0")
            success_chars.append(char)
        except Exception as e:
            print(f"Error while preprocessing dataset for {char}: {e}")
            fail_chars.append(char)

    if fail_chars:
        print(f"Dataset preprocessing (name_to_semantic) completed with errors for characters: {fail_chars}. Success for: {success_chars}")
        return False
    else:
        print(f"Dataset preprocessing (name_to_semantic) completed successfully for characters: {success_chars}")
        return True


def train_model_gpt_cli(batch_size: int = None, total_epoch: int = 15):
    before_request() # to load config.muted_characters
    success_chars = []
    fail_chars = []
    for char in config.muted_characters:
        try:
            if char in config.models_path and config.models_path[char][0] != "":
                print(f"GPT model already exists for {char}, skipping training.")
                success_chars.append(char) # skip if model already exists
                continue
            communication.train_s1(char, "0", batch_size=batch_size, total_epoch=total_epoch)
            success_chars.append(char)
        except Exception as e:
            print(f"Error while training GPT model for {char}: {e}")
            fail_chars.append(char)

    if fail_chars:
        print(f"GPT model training completed with errors for characters: {fail_chars}. Success for: {success_chars}")
        return False
    else:
        print(f"GPT model training completed successfully for characters: {success_chars}")
        return True


def train_model_sovits_cli(batch_size: int = None, total_epoch: int = 15):
    before_request() # to load config.muted_characters
    success_chars = []
    fail_chars = []
    for char in config.muted_characters:
        try:
            if char in config.models_path and config.models_path[char][1] != "":
                print(f"SoViTS model already exists for {char}, skipping training.")
                success_chars.append(char) # skip if model already exists
                continue
            communication.train_s2(char, "0", batch_size=batch_size, total_epoch=total_epoch)
            success_chars.append(char)
        except Exception as e:
            print(f"Error while training SoViTS model for {char}: {e}")
            fail_chars.append(char)

    if fail_chars:
        print(f"SoViTS model training completed with errors for characters: {fail_chars}. Success for: {success_chars}")
        return False
    else:
        print(f"SoViTS model training completed successfully for characters: {success_chars}")
        return True


def info_cli():
    before_request()
    info_data = {
        "status": "running",
        "models_path": config.models_path,
        "available_characters": config.muted_characters,
    }
    print("Status: running")
    print("Models Path:", info_data["models_path"])
    print("Available Characters:", info_data["available_characters"])
    return True


def main():
    parser = argparse.ArgumentParser(description="AIDub CLI Management Tool")
    subparsers = parser.add_subparsers(title="commands", dest="command")

    # download_dataset
    download_dataset_parser = subparsers.add_parser("download_dataset", help="Download dataset for specified characters")
    download_dataset_parser.add_argument("--char-names", type=str, help="Comma-separated character names", required=True)
    download_dataset_parser.add_argument("--sources-to-fetch", type=str, help="Comma-separated data sources to fetch voice from", required=True)

    # emotion_classification
    emotion_classification_parser = subparsers.add_parser("emotion_classification", help="Perform emotion classification")

    # dub
    dub_parser = subparsers.add_parser("dub", help="Dub text using specified character")
    dub_parser.add_argument("--text", type=str, help="Text to dub", required=True)
    dub_parser.add_argument("--char-name", type=str, help="Character name for dubbing", required=True)

    # preprocess_dataset subparsers
    preprocess_dataset_parser = subparsers.add_parser("preprocess_dataset", help="Dataset preprocessing operations")
    preprocess_subparsers = preprocess_dataset_parser.add_subparsers(title="preprocess_commands", dest="preprocess_command")

    # preprocess_dataset get_text
    preprocess_subparsers.add_parser("get_text", help="Preprocess dataset: get_text")

    # preprocess_dataset get_hubert_wav32k
    preprocess_subparsers.add_parser("get_hubert_wav32k", help="Preprocess dataset: get_hubert_wav32k")

    # preprocess_dataset name_to_semantic
    preprocess_subparsers.add_parser("name_to_semantic", help="Preprocess dataset: name_to_semantic")


    # train_model subparsers
    train_model_parser = subparsers.add_parser("train_model", help="Model training operations")
    train_subparsers = train_model_parser.add_subparsers(title="train_commands", dest="train_command")

    # train_model gpt
    train_gpt_parser = train_subparsers.add_parser("gpt", help="Train GPT model")
    train_gpt_parser.add_argument("--batch-size", type=int, help="Batch size for training")
    train_gpt_parser.add_argument("--total-epoch", type=int, default=15, help="Total epochs for training (default: 15)")

    # train_model sovits
    train_sovits_parser = train_subparsers.add_parser("sovits", help="Train SoViTS model")
    train_sovits_parser.add_argument("--batch-size", type=int, help="Batch size for training")
    train_sovits_parser.add_argument("--total-epoch", type=int, default=15, help="Total epochs for training (default: 15)")

    # info
    info_parser = subparsers.add_parser("info", help="Get AIDub status information")

    args = parser.parse_args()

    if args.command == "download_dataset":
        char_names = [name.strip() for name in args.char_names.split(',')]
        sources_to_fetch = [source.strip() for source in args.sources_to_fetch.split(',')]
        download_dataset_cli(char_names, sources_to_fetch)
    elif args.command == "emotion_classification":
        emotion_classification_cli()
    elif args.command == "dub":
        dub_cli(args.text, args.char_name)
    elif args.command == "preprocess_dataset":
        if args.preprocess_command == "get_text":
            get_text_cli()
        elif args.preprocess_command == "get_hubert_wav32k":
            get_hubert_wav32k_cli()
        elif args.preprocess_command == "name_to_semantic":
            name_to_semantic_cli()
    elif args.command == "train_model":
        if args.train_command == "gpt":
            train_model_gpt_cli(batch_size=args.batch_size, total_epoch=args.total_epoch)
        elif args.train_command == "sovits":
            train_model_sovits_cli(batch_size=args.batch_size, total_epoch=args.total_epoch)
    elif args.command == "info":
        info_cli()
    elif args.command is None:
        parser.print_help()
    else:
        print(f"Unknown command: {args.command}")


if __name__ == '__main__':
    main()