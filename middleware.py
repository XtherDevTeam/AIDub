import os
import pathlib
import flask_cors
import flask
import typing
import config
import common
import fandom
import voice_fetch
import emotion
import dub
import communication

def run_gpt_sovits_server():
    os.system("cd GPT-SoVITs && python api_v2.py")


app = flask.Flask(__name__)

def makeResult(ok: bool = True, data: typing.Any = None) -> dict:
    return {
        'status': ok,
        'data': data
    }

def do_voice_collection():
    collections: dict[str, list[tuple[str, str]]] = {}
    
    for i in config.sources_to_fetch_voice:
        
        quests = common.request_retry_wrapper(lambda: fandom.fetch_quest_entries(i))
        

        for quest in quests:
            collection = fandom.fetch_target_vo_from_quest_page(quest, config.muted_characters)

            collections = fandom.merge_voice_collections([collections, collection])

    # download voices
    voice_fetch.reduce_collection(collections)
    voice_fetch.fetch_collection(collections)

    with open(config.dataset_manifest_file_dest, 'w') as f:
        f.write(voice_fetch.serialize_collection(collections))

    voice_fetch.generate_text_list()


@app.route('/download_dataset', methods=['POST'])
def download_dataset():
    request_form = flask.request.json
    char_names = request_form['char_names']
    config.muted_characters = char_names
    do_voice_collection()
    
    return makeResult(ok=True, data=f"Dataset for {char_names} downloaded successfully.")



@app.route('/emotion_classification', methods=['POST'])
def emotion_classification():
    emotion.do_classification()
    return makeResult(ok=True, data=f'')


@app.route('/dub')
def dub_route():
    form = flask.request.json
    text = form['text']
    char_name = form['char_name']
    d = emotion.choose_a_voice_by_text(char_name, text)
    r = dub.dub_one(text, char_name, True)
    resp = flask.make_response()
    resp.headers['Content-Type'] = 'audio/aac'
    resp.headers['Content-Disposition'] = f'attachment; filename="{char_name}.aac"'
    resp.data = r
    return resp


@app.route('/gpt_sovits/dataset_preprocessing/get_text')
def get_text():
    for char in config.muted_characters:
        try:
            dest = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}")
            list = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}.list")
            communication.preprocess_dataset("1a", char, list.absolute(), dest.absolute(), "0-1")
        except Exception as e:
            return makeResult(ok=False, data=f"Error while preprocessing dataset for {char}: {e}")
    return makeResult(ok=True, data=f"Dataset for {config.muted_characters} preprocessed successfully.")

@app.route('/gpt_sovits/dataset_preprocessing/get_hubert_wav32k')
def get_hubert_wav32k():
    for char in config.muted_characters:
        try:
            dest = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}")
            list = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}.list")
            communication.preprocess_dataset("1b", char, list.absolute(), dest.absolute())
        except Exception as e:
            return makeResult(ok=False, data=f"Error while preprocessing dataset for {char}: {e}")
    return makeResult(ok=True, data=f"Dataset for {config.muted_characters} preprocessed successfully.")

@app.route('/gpt_sovits/dataset_preprocessing/name_to_semantic')
def name_to_semantic():
    for char in config.muted_characters:
        try:
            dest = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}")
            list = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}.list")
            communication.preprocess_dataset("1c", char, list.absolute(), dest.absolute())
        except Exception as e:
            return makeResult(ok=False, data=f"Error while preprocessing dataset for {char}: {e}")
    return makeResult(ok=True, data=f"Dataset for {config.muted_characters} preprocessed successfully.")


@app.route('/gpt_sovits/train_model_gpt')
def train_model_gpt():
    form = flask.request.json
    batch_size = form.get('batch_size', None)
    total_epoch = form.get('total_epoch', 15)
    for char in config.muted_characters:
        try:
            communication.train_s1(char, "0-1", batch_size=batch_size, total_epoch=total_epoch)
            return makeResult(ok=True, data=f"Model for {char} trained successfully.")
        except Exception as e:
            return makeResult(ok=False, data=f"Error while training model for {char}: {e}")


@app.route('/gpt_sovits/train_model_sovits')
def train_model_sovits():
    form = flask.request.json
    batch_size = form.get('batch_size', None)
    total_epoch = form.get('total_epoch', 15)
    for char in config.muted_characters:
        try:
            communication.train_s2(char, "0-1", batch_size=batch_size, total_epoch=total_epoch)
            return makeResult(ok=True, data=f"Model for {char} trained successfully.")
        except Exception as e:
            return makeResult(ok=False, data=f"Error while training model for {char}: {e}")
    