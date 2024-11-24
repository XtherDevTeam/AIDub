"""
https://github.com/XtherDevTeam/AIDub

AIDub Middleware

Copyright (C) 2024 Jerry Chou, All rights reserved.
Open-source under the MIT license.

This module implements the API server for AIDub for easier integation with other applications,
includes the following functionalities:

- Dataset downloading
- Emotion classification
- Dubbing
- GPT-SoVITs dataset preprocessing
- GPT-SoVITs model training

To run the server, simply run the following command in the terminal:

```
python middleware.py
```

The server will be running on `http://localhost:2731`.
"""


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
import importlib
import communication

def run_gpt_sovits_server():
    os.system("cd GPT-SoVITs && python api_v2.py")


app = flask.Flask(__name__)

@app.before_request
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


@app.route('/download_dataset', methods=['POST'])
def download_dataset():
    request_form = flask.request.json
    char_names = request_form.get('char_names', [])
    sources_to_fetch = request_form.get('sources_to_fetch', [])
    if char_names == [] or sources_to_fetch == []:
        return makeResult(ok=False, data=f"Invalid request, char_names and sources_to_fetch are required.")
    
    config.muted_characters = [i for i in set(char_names + config.muted_characters)]
    config.sources_to_fetch_voice = [i for i in set(sources_to_fetch + config.sources_to_fetch_voice)]
    do_voice_collection()
    
    return makeResult(ok=True, data=f"Dataset for {char_names} downloaded successfully.")



@app.route('/emotion_classification', methods=['POST'])
def emotion_classification():
    emotion.do_classification()
    return makeResult(ok=True, data=f'')


@app.route('/dub', methods=['POST', 'GET'])
def dub_route():
    if flask.request.method == 'POST' or flask.request.headers.get('Content-Type') == 'application/json':
        form = flask.request.json
        text = form['text']
        char_name = form['char_name']
        ckpt, pth = dub.get_tts_models(char_name)
        dub.setup_gpt_sovits_client(ckpt, pth)
        r = dub.dub_one(text, char_name, True)
        # resp = flask.make_response()
        # resp.headers['Content-Type'] = r.headers['Content-Type']
        # resp.data = r.iter_content(chunk_size=10*1024)
        # return resp
        headers = r.headers
        headers['Content-Disposition'] = f'attachment; filename={char_name}.aac'
        return flask.Response(r.iter_content(chunk_size=10*1024), headers=headers)
    else:
        # check if body is application/json
        text = flask.request.args.get('text')
        char_name = flask.request.args.get('char_name')
        ckpt, pth = dub.get_tts_models(char_name)
        dub.setup_gpt_sovits_client(ckpt, pth)
        r = dub.dub_one(text, char_name, True)
        # resp = flask.make_response()
        # resp.headers['Content-Type'] = r.headers['Content-Type']
        # resp.data = r.iter_content(chunk_size=10*1024)
        # return resp
        headers = r.headers
        headers['Content-Disposition'] = f'attachment; filename={char_name}.aac'
        return flask.Response(r.iter_content(chunk_size=10*1024), headers=headers)


@app.route('/gpt_sovits/dataset_preprocessing/get_text', methods=['POST'])
def get_text():
    for char in config.muted_characters:
        try:
            dest = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}")
            list = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}/{char}.list")
            communication.preprocess_dataset("1a", char, str(list.absolute()), str(dest.absolute()), "0")
        except Exception as e:
            return makeResult(ok=False, data=f"Error while preprocessing dataset for {char}: {e}")
    return makeResult(ok=True, data=f"Dataset for {config.muted_characters} preprocessed successfully.")

@app.route('/gpt_sovits/dataset_preprocessing/get_hubert_wav32k', methods=['POST'])
def get_hubert_wav32k():
    for char in config.muted_characters:
        try:
            dest = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}")
            list = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}/{char}.list")
            communication.preprocess_dataset("1b", char, str(list.absolute()), str(dest.absolute()), "0")
        except Exception as e:
            return makeResult(ok=False, data=f"Error while preprocessing dataset for {char}: {e}")
    return makeResult(ok=True, data=f"Dataset for {config.muted_characters} preprocessed successfully.")

@app.route('/gpt_sovits/dataset_preprocessing/name_to_semantic', methods=['POST'])
def name_to_semantic():
    for char in config.muted_characters:
        try:
            dest = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}")
            list = pathlib.Path(f"{config.save_dest_for_downloaded_voice}/{char}/{char}.list")
            communication.preprocess_dataset("1c", char, str(list.absolute()), str(dest.absolute()), "0")
        except Exception as e:
            return makeResult(ok=False, data=f"Error while preprocessing dataset for {char}: {e}")
    return makeResult(ok=True, data=f"Dataset for {config.muted_characters} preprocessed successfully.")


@app.route('/gpt_sovits/train_model_gpt', methods=['POST'])
def train_model_gpt():
    form = flask.request.json
    batch_size = form.get('batch_size', None)
    total_epoch = form.get('total_epoch', 15)
    for char in config.muted_characters:
        try:
            if char in config.models_path:
                continue
            communication.train_s1(char, "0", batch_size=batch_size, total_epoch=total_epoch)
        except Exception as e:
            return makeResult(ok=False, data=f"Error while training model for {char}: {e}")
    return makeResult(ok=True, data=f"Model for {config.muted_characters} trained successfully.")


@app.route('/gpt_sovits/train_model_sovits', methods=['POST'])
def train_model_sovits():
    form = flask.request.json
    batch_size = form.get('batch_size', None)
    total_epoch = form.get('total_epoch', 15)
    for char in config.muted_characters:
        try:
            if char in config.models_path:
                continue
            communication.train_s2(char, "0", batch_size=batch_size, total_epoch=total_epoch)
        except Exception as e:
            return makeResult(ok=False, data=f"Error while training model for {char}: {e}")
        
    return makeResult(ok=True, data=f"Model for {config.muted_characters} trained successfully.")
    
    
@app.route('/info', methods=['POST'])
def info():
    return makeResult(ok=True, data={
        "status": "running",
        "models_path": config.models_path,
        "available_characters": config.muted_characters,
    })
    

if __name__ == '__main__':
    # dynamically load available models and muted characters, differ from the ones in config.py
    app.run(debug=False, host='192.168.1.7', port=2731)