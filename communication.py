import os
import json
import yaml
from thirdparty.GPTSoViTs.config import python_exec, exp_root, is_half, pretrained_sovits_path
from thirdparty.GPTSoViTs.tools.my_utils import clean_path, check_for_existance, check_details
from subprocess import Popen, PIPE
import sys
from thirdparty.GPTSoViTs.tools import my_utils
import config

SoVITS_weight_root=["GPTSoViTs/SoVITS_weights_v2","GPTSoViTs/SoVITS_weights"]
GPT_weight_root=["GPTSoViTs/GPT_weights_v2","GPTSoViTs/GPT_weights"]


def gpu_numbers():
    """
    获取 GPU 卡号.
    """
    gpu_nums = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu_nums == "":
        gpu_nums = "0"
    return gpu_nums


def preprocess_dataset(step: str, exp_name, inp_text, inp_wav_dir, gpu_numbers, bert_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large", ssl_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-hubert-base"):
    """
    对数据集进行预处理，生成配置文件并训练 GPT 模型.

    Args:
        step (str): 预处理步骤，目前支持 "1a"、"1b"、"1c"
        exp_name (str): 实验名称.
        inp_text (str): 文本标注文件路径.
        inp_wav_dir (str): 音频数据集路径.
        gpu_numbers (str): GPU 卡号，以 '-' 分割.
        bert_pretrained_dir (str, optional): 预训练 BERT 模型路径. Defaults to "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large".
        ssl_pretrained_dir (str, optional): 预训练 SSL 模型路径. Defaults to "GPT_SoVITS/pretrained_models/chinese-hubert-base".
    """
    inp_text = clean_path(inp_text)
    inp_wav_dir = clean_path(inp_wav_dir)

    if check_for_existance([inp_text, inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text, inp_wav_dir], is_dataset_processing=True)

    opt_dir = os.path.join('thirdparty', 'GPTSoViTs', exp_root, exp_name)
    passing_opt_dir = os.path.join(exp_root, exp_name)


    def run_process(script, env_vars):
        """运行子进程."""
        os.environ.update(env_vars)
        cmd = f'"{python_exec}" {script}'
        print(cmd)
        p = Popen(cmd, shell=True, stdout=sys.stdout.fileno(), stderr=sys.stderr.fileno(), cwd="thirdparty/GPTSoViTs")
        p.communicate()


    if step == "1a":
        # 1a. 文本处理
        path_text = os.path.join(opt_dir, "2-name2text.txt")
        if not os.path.exists(path_text):
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": passing_opt_dir,
                "bert_pretrained_dir": bert_pretrained_dir,
                "is_half": str(is_half),
            }
            gpu_names = gpu_numbers.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update({
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                })
                run_process("GPT_SoVITS/prepare_datasets/1-get-text.py", config)
                print(f"1a done for part {i_part} of {all_parts} parts, exp_name: {exp_name} with config {config}")

            opt = []
            for i_part in range(all_parts):
                txt_path = os.path.join(opt_dir, f"2-name2text-{i_part}.txt")
                with open(txt_path, "r", encoding="utf8") as f:
                    opt += f.read().strip("\n").split("\n")
                os.remove(txt_path)
            with open(path_text, "w", encoding="utf8") as f:
                f.write("\n".join(opt) + "\n")
            assert len("".join(opt)) > 0, "1a-文本获取进程失败"
            return True
        

    elif step == "1b":
        # 1b. SSL 特征提取
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": passing_opt_dir,
            "cnhubert_base_dir": ssl_pretrained_dir,
            "is_half": str(is_half),
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update({
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
            })
            run_process("GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py", config)
            return True

    elif step == "1c":
        # 1c. 语义 token 提取
        path_semantic = os.path.join(opt_dir, "6-name2semantic.tsv")
        if not os.path.exists(path_semantic) or os.path.getsize(path_semantic) < 31:
            config = {
                "inp_text": inp_text,
                "exp_name": exp_name,
                "opt_dir": passing_opt_dir,
                "pretrained_s2G": SoVITS_weight_root[-int("v2"[-1]) + 2], #版本写死v2
                "s2config_path": "GPT_SoVITS/configs/s2.json",
                "is_half": str(is_half),
            }
            gpu_names = gpu_numbers.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update({
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                })
                run_process("GPT_SoVITS/prepare_datasets/3-get-semantic.py", config)

            opt = ["item_name\tsemantic_audio"]
            for i_part in range(all_parts):
                semantic_path = os.path.join(opt_dir, f"6-name2semantic-{i_part}.tsv")
                with open(semantic_path, "r", encoding="utf8") as f:
                    opt += f.read().strip("\n").split("\n")
                os.remove(semantic_path)
            with open(path_semantic, "w", encoding="utf8") as f:
                f.write("\n".join(opt) + "\n")
                return True

    else:
        raise RuntimeError("不支持的预处理步骤")
    

def train_s1(exp_name, gpu_numbers, batch_size=None, total_epoch=15, if_dpo=False, if_save_latest=True, if_save_every_weights=True, save_every_epoch=5, pretrained_s1=None, version="v2"):
    """
    预处理数据集并训练 S1 (GPT) 模型.
    Args:
        exp_name (str): 实验名称.
        gpu_numbers (str): GPU 卡号，以 '-' 分割.
        bert_pretrained_dir (str, optional): 预训练 BERT 模型路径. Defaults to "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large".
        ssl_pretrained_dir (str, optional): 预训练 SSL 模型路径. Defaults to "GPT_SoVITS/pretrained_models/chinese-hubert-base".
        batch_size (int, optional): 训练 batch size. Defaults to None.
        total_epoch (int, optional): 训练 epoch 数. Defaults to 15.
        if_dpo (bool, optional): 是否使用 DPO 优化器. Defaults to False.
        if_save_latest (bool, optional): 是否保存最新模型. Defaults to True.
        if_save_every_weights (bool, optional): 是否保存每一个 epoch 的模型. Defaults to True.
        save_every_epoch (int, optional): 保存模型的间隔 epoch 数. Defaults to 5.
        pretrained_s1 (str, optional): 预训练 S1 模型路径. Defaults to None.
        version (str, optional): 模型版本. Defaults to "v2".
    """

    with open("GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml") as f:
        data = yaml.safe_load(f)

    if is_half == False:
        data["train"]["precision"] = "32"
        batch_size = max(1, batch_size // 2) if batch_size is not None else None # 根据is_half调整batch_size

    data["train"]["batch_size"] = batch_size # 传入参数
    data["train"]["epochs"] = total_epoch
    data["pretrained_s1"] = pretrained_s1
    data["train"]["save_every_n_epoch"] = save_every_epoch
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_dpo"] = if_dpo
    data["train"]["half_weights_save_dir"] = GPT_weight_root[-int(version[-1]) + 2]
    data["train"]["exp_name"] = exp_name
    opt_dir = os.path.join('thirdparty', 'GPTSoViTs', exp_root, exp_name)
    passing_opt_dir = os.path.join(exp_root, exp_name)
    data["train_semantic_path"] = os.path.join(passing_opt_dir, "6-name2semantic.tsv")
    data["train_phoneme_path"] = os.path.join(passing_opt_dir, "2-name2text.txt")
    data["output_dir"] = os.path.join(passing_opt_dir, "logs_s1")

    tmp_config_path = os.path.join('thirdparty', 'GPTSoViTs', "TEMP", "tmp_s1.yaml")
    with open(tmp_config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    os.environ["_CUDA_VISIBLE_DEVICES"] = ",".join(gpu_numbers.split("-"))
    os.environ["hz"] = "25hz"
    cmd = f'"{python_exec}" GPT_SoVITS/s1_train.py --config_file "{tmp_config_path}"'
    print("GPT训练开始：", cmd)
    p = Popen(cmd, shell=True)
    p.wait()
    print("GPT训练完成")


def train_s2(exp_name, gpu_numbers, batch_size=None, total_epoch=8, text_low_lr_rate=0.4, if_save_latest=True, if_save_every_weights=True, save_every_epoch=4, pretrained_s2G=None, pretrained_s2D=None, version="v2"):
    """
    训练 S2 (SoVITS) 模型.
    Args:
        exp_name (str): 实验名称.
        gpu_numbers (str): GPU 卡号，以 '-' 分割.
        batch_size (int, optional): 训练 batch size. Defaults to None.
        total_epoch (int, optional): 训练 epoch 数. Defaults to 8.
        text_low_lr_rate (float, optional): 文本低学习率. Defaults to 0.4.
        if_save_latest (bool, optional): 是否保存最新模型. Defaults to True.
        if_save_every_weights (bool, optional): 是否保存每一个 epoch 的模型. Defaults to True.
        save_every_epoch (int, optional): 保存模型的间隔 epoch 数. Defaults to 4.
        pretrained_s2G (str, optional): 预训练 S2G 模型路径. Defaults to None.
        pretrained_s2D (str, optional): 预训练 S2D 模型路径. Defaults to None.
        version (str, optional): 模型版本. Defaults to "v2".
    """


    with open("GPT_SoVITS/configs/s2.json") as f:
        data = json.load(f)

    s2_dir = os.path.join('thirdparty', 'GPTSoViTs', exp_root, exp_name)
    passing_s2_dir = os.path.join(exp_root, exp_name)
    os.makedirs(os.path.join(s2_dir, "logs_s2"), exist_ok=True)

    if check_for_existance([s2_dir], is_train=True):
        check_details([s2_dir], is_train=True)

    if is_half == False:
        data["train"]["fp16_run"] = False
        batch_size = max(1, batch_size // 2) if batch_size is not None else None

    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["train"]["text_low_lr_rate"] = text_low_lr_rate
    data["train"]["pretrained_s2G"] = pretrained_s2G
    data["train"]["pretrained_s2D"] = pretrained_s2D
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["save_every_epoch"] = save_every_epoch
    data["train"]["gpu_numbers"] = gpu_numbers  # 这里不需要分割
    data["model"]["version"] = version
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = passing_s2_dir
    data["save_weight_dir"] = SoVITS_weight_root[-int(version[-1]) + 2]
    data["name"] = exp_name
    data["version"] = version

    tmp_config_path = os.path.join('thirdparty', 'GPTSoViTs', "TEMP", "tmp_s2.json")
    with open(tmp_config_path, "w") as f:
        json.dump(data, f)

    cmd = f'"{python_exec}" GPT_SoVITS/s2_train.py --config "{tmp_config_path}"'
    print("SoVITS训练开始：", cmd)
    p = Popen(cmd, shell=True)
    p.wait()
    print("SoVITS训练完成")
