import pathlib
import random
import time

import config
import json
import os

def start():
    print("Preparing dataset...")
    download_model()
    dataset_manifest = json.loads(pathlib.Path(config.dataset_manifest_file_dest).read_text())
    for char in dataset_manifest:
        for label in dataset_manifest[char]:
            item = dataset_manifest[char][label]
            path = item['dest']
            text = item['text']
            with open(pathlib.Path(path).with_suffix('.lab'), 'w+') as f:
                f.write(text)
        extract_vq(char)
        proto_path = packup_dataset(char)
        t2s_config = generate_t2s_config(proto_path)
        finetune_model(t2s_config, char)


def download_model():
    print("Downloading model...")
    os.system(f"huggingface-cli download fishaudio/fish-speech-1.4 --local-dir {config.fish_speech_module_path}/checkpoints/fish-speech-1.4")

def extract_vq(char: str):
    print(f"Extracting VQ for {char}...")
    data = pathlib.Path(config.save_dest_for_downloaded_voice) / char
    os.system(f"python {config.fish_speech_module_path}/tools/vqgan/extract_vq.py \"{data}\" \
    --num-workers 1 --batch-size 16 \
    --config-name \"firefly_gan_vq\" \
    --checkpoint-path \"{config.fish_speech_module_path}/checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth\"")

def packup_dataset(char: str):
    print(f"Packing up dataset for {char}...")
    data = pathlib.Path(config.save_dest_for_downloaded_voice) / char
    protos = data / "protos"
    os.system(f"python {config.fish_speech_module_path}/tools/llama/build_dataset.py \
    --input \"{data}\" \
    --output \"{protos}\" \
    --text-extension .lab \
    --num-workers 16")
    return str(protos)

def generate_t2s_config(proto_path: str):
    name = f"t2s_config_{random.randint(0, 1048576)}"
    temp = f"{config.fish_speech_module_path}/fish_speech/configs/{name}.yaml"
    template = pathlib.Path(f"{config.fish_speech_module_path}/fish_speech/configs/text2semantic_finetune.yaml").read_text()
    with open(temp, 'w+') as f:
        f.write(template
                .replace("data/protos", f"{proto_path}")
                .replace("checkpoints/fish-speech-1.4", f"{config.fish_speech_module_path}/checkpoints/fish-speech-1.4"))
    return name
    

def finetune_model(temp_config: str, project: str):
    print("Fine-tuning model...")
    
    os.system(f"python {config.fish_speech_module_path}/fish_speech/train.py --config-name {temp_config} \
    project={project} \
    +lora@model.model.lora_config=r_8_alpha_16")