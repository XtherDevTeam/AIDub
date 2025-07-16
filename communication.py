import os
import json
import yaml
from subprocess import Popen, PIPE, STDOUT
import sys
import shutil
import traceback  # Import traceback for better error info if needed

# --- Configuration (mimicking parts of webui.py and config.py) ---
# Assuming this script is in the project root, and GPTSoViTs is in 'thirdparty/GPTSoViTs'
# Corrected path to GPTSoVITS
GPT_SOVITS_DIR = os.path.join("thirdparty", "GPTSoVITs")
NOW_DIR = os.getcwd()

# Get python executable path from original webui.py config.py if possible, otherwise use sys.executable
# Assuming config.py is in the GPTSoVITS root or accessible
try:
    sys.path.insert(0, GPT_SOVITS_DIR)
    # webui might use a specific env
    from config import python_exec as PYTHON_EXEC_WEBUI
    from config import is_half as IS_HALF_WEBUI
    from config import exp_root as EXP_ROOT_WEBUI
    from config import pretrained_sovits_name as PRETRAINED_SOVITS_NAME_WEBUI_MAP
    from config import pretrained_gpt_name as PRETRAINED_GPT_NAME_WEBUI_MAP
    from config import SoVITS_weight_version2root as SOVITS_WEIGHT_ROOT_MAP_WEBUI
    from config import GPT_weight_version2root as GPT_WEIGHT_ROOT_MAP_WEBUI
    sys.path.pop(0)  # Remove temporarily added path
    PYTHON_EXEC = sys.executable # Use sys.executable in default
    IS_HALF = IS_HALF_WEBUI
    EXP_ROOT = EXP_ROOT_WEBUI  # Use webui's exp_root definition
    PRETRAINED_SOVITS_NAME_MAP = PRETRAINED_SOVITS_NAME_WEBUI_MAP
    PRETRAINED_GPT_NAME_MAP = PRETRAINED_GPT_NAME_WEBUI_MAP
    SOVITS_WEIGHT_ROOT_MAP = SOVITS_WEIGHT_ROOT_MAP_WEBUI
    GPT_WEIGHT_ROOT_MAP = GPT_WEIGHT_ROOT_MAP_WEBUI

    print("Loaded configuration from thirdparty/GPTSoVITS/config.py")

except ImportError:
    print("Warning: Could not import config from thirdparty/GPTSoVITS/. Using default settings.")
    # Fallback if config.py import fails
    PYTHON_EXEC = sys.executable
    IS_HALF = True
    EXP_ROOT = "logs"  # Relative to GPT_SOVITS_DIR

    # Define default pretrained model maps if config.py not found
    PRETRAINED_BASE_DIR_DEFAULT = os.path.join(
        "GPT_SoVITS", "pretrained_models")
    PRETRAINED_SOVITS_NAME_MAP = {
        "v1": os.path.join(PRETRAINED_BASE_DIR_DEFAULT, "s2G488k.pth"),
        "v2": os.path.join(PRETRAINED_BASE_DIR_DEFAULT, "s2G488k.pth"),
        "v2Pro": os.path.join(PRETRAINED_BASE_DIR_DEFAULT, "v2Pro", "s2Gv2Pro.pth"),
        "v2ProPlus": os.path.join(PRETRAINED_BASE_DIR_DEFAULT, "v2Pro", "s2Gv2ProPlus.pth"),
        "v3": os.path.join(PRETRAINED_BASE_DIR_DEFAULT, "s2Gv3.pth"),
        "v4": os.path.join(PRETRAINED_BASE_DIR_DEFAULT, "gsv-v4-pretrained", "s2Gv4.pth"),
    }
    PRETRAINED_GPT_NAME_MAP = {
        "v1": os.path.join(PRETRAINED_BASE_DIR_DEFAULT, "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"),
        "v2": os.path.join(PRETRAINED_BASE_DIR_DEFAULT, "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
        "v2Pro": os.path.join(PRETRAINED_BASE_DIR_DEFAULT, "s1v3.ckpt"),
        "v2ProPlus": os.path.join(PRETRAINED_BASE_DIR_DEFAULT, "s1v3.ckpt"),
        "v3": os.path.join(PRETRAINED_BASE_DIR_DEFAULT, "s1v3.ckpt"),
        "v4": os.path.join(PRETRAINED_BASE_DIR_DEFAULT, "s1v3.ckpt"),
    }
    # Define default weight root maps if config.py not found
    _SOVITS_WEIGHT_ROOT_DEFAULT = "SoVITS_weights"
    _SOVITS_WEIGHT_ROOT_V2_SUFFIXED = "SoVITS_weights_v2"
    _GPT_WEIGHT_ROOT_DEFAULT = "GPT_weights"
    _GPT_WEIGHT_ROOT_V2_SUFFIXED = "GPT_weights_v2"

    SOVITS_WEIGHT_ROOT_MAP = {
        "v1": _SOVITS_WEIGHT_ROOT_DEFAULT,
        "v2": _SOVITS_WEIGHT_ROOT_DEFAULT,
        "v2Pro": _SOVITS_WEIGHT_ROOT_V2_SUFFIXED,
        "v2ProPlus": _SOVITS_WEIGHT_ROOT_V2_SUFFIXED,
        "v3": _SOVITS_WEIGHT_ROOT_DEFAULT,
        "v4": _SOVITS_WEIGHT_ROOT_DEFAULT,
    }
    GPT_WEIGHT_ROOT_MAP = {
        "v1": _GPT_WEIGHT_ROOT_DEFAULT,
        "v2": _GPT_WEIGHT_ROOT_DEFAULT,
        "v2Pro": _GPT_WEIGHT_ROOT_V2_SUFFIXED,
        "v2ProPlus": _GPT_WEIGHT_ROOT_V2_SUFFIXED,
        "v3": _GPT_WEIGHT_ROOT_DEFAULT,
        "v4": _GPT_WEIGHT_ROOT_DEFAULT,
    }

# Helper to get SoVITS-D from SoVITS-G path


def get_s2d_path(s2g_path):
    if not s2g_path:
        return ""
    return s2g_path.replace("s2G", "s2D")


# Define default pretrained paths based on a default version (e.g., v2)
# These are just examples; the functions will use the maps based on the 'version' parameter.
DEFAULT_BERT_PRETRAINED_DIR = os.path.join(PRETRAINED_BASE_DIR_DEFAULT if 'PRETRAINED_BASE_DIR_DEFAULT' in globals(
) else os.path.join("GPT_SoVITS", "pretrained_models"), "chinese-roberta-wwm-ext-large")
DEFAULT_SSL_PRETRAINED_DIR = os.path.join(PRETRAINED_BASE_DIR_DEFAULT if 'PRETRAINED_BASE_DIR_DEFAULT' in globals(
) else os.path.join("GPT_SoVITS", "pretrained_models"), "chinese-hubert-base")
DEFAULT_SV_PATH = os.path.join(PRETRAINED_BASE_DIR_DEFAULT if 'PRETRAINED_BASE_DIR_DEFAULT' in globals(
) else os.path.join("GPT_SoVITS", "pretrained_models"), "sv", "pretrained_eres2netv2w24s4ep4.ckpt")


V3_V4_SET = {"v3", "v4"}
PRO_SET = {"v2Pro", "v2ProPlus"}  # Added for step 1b logic

# Ensure TEMP directory exists within GPT_SOVITS_DIR
GPT_SOVITS_TEMP_DIR = os.path.join(GPT_SOVITS_DIR, "TEMP")
os.makedirs(GPT_SOVITS_TEMP_DIR, exist_ok=True)

# Create weight directories based on the maps
for r_path in set(list(SOVITS_WEIGHT_ROOT_MAP.values()) + list(GPT_WEIGHT_ROOT_MAP.values())):
    os.makedirs(os.path.join(GPT_SOVITS_DIR, r_path), exist_ok=True)


def _clean_path(path):
    if path is None:
        return None
    return path.strip().strip('"').strip("'").strip()


def _get_gpu_numbers_str_hyphenated(gpu_indices_str):
    # Input is expected to be like "0", "0-1", "0,1"
    if not isinstance(gpu_indices_str, str):
        # Handle int/list accidentally passed
        return str(gpu_indices_str).replace(",", "-")
    return gpu_indices_str.replace(",", "-")  # Ensure hyphenated


def _get_gpu_numbers_str_commaed(gpu_indices_str):
    # Input is expected to be like "0", "0-1", "0,1"
    if not isinstance(gpu_indices_str, str):
        # Handle int/list accidentally passed
        return str(gpu_indices_str).replace("-", ",")
    return gpu_indices_str.replace("-", ",")  # Ensure commaed


def _run_subprocess(cmd_list, env_vars=None, cwd=GPT_SOVITS_DIR):
    current_env = os.environ.copy()
    if env_vars:
        for k, v in env_vars.items():
            current_env[k] = str(v)

    # Use the python executable path loaded from config or sys.executable
    executable_cmd_list = [PYTHON_EXEC] + cmd_list

    print(f"Running command: {' '.join(executable_cmd_list)}")
    print(f"With CWD: {cwd}")
    if env_vars:
        print(f"With extra ENV: {env_vars}")

    # Popen with PIPE and reading stdout/stderr is generally more robust
    # than redirecting to sys.stdout/stderr, especially across platforms.
    # The original snippet used sys.stdout.fileno(), which is simpler
    # but might have buffering issues or platform differences.
    # Let's stick to reading PIPE for better control and logging.
    # Use bufsize=1 for line-buffered output
    process = Popen(executable_cmd_list, env=current_env, cwd=cwd,
                    stdout=PIPE, stderr=STDOUT, universal_newlines=True, bufsize=1)
    for line in process.stdout:
        print(line.strip())
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed with return code {process.returncode}: {' '.join(executable_cmd_list)}")
    print(f"Subprocess completed: {' '.join(executable_cmd_list)}")


# --- Dataset Formatting Function (1A, 1B, 1C) ---

def preprocess_dataset(step: str, exp_name: str, inp_text: str, inp_wav_dir: str, gpu_numbers: str,
                       bert_pretrained_dir: str = DEFAULT_BERT_PRETRAINED_DIR,
                       ssl_pretrained_dir: str = DEFAULT_SSL_PRETRAINED_DIR,
                       version: str = "v2Pro"):  # Added version parameter
    """
    Dataset preprocessing steps 1a, 1b, or 1c.

    Args:
        step (str): Preprocessing step, e.g., "1a", "1b", "1c".
        exp_name (str): Experiment name.
        inp_text (str): Input text list path (absolute or relative to NOW_DIR).
                         For 1a: The initial list file.
                         For 1b, 1c: The 2-name2text.txt file generated by 1a
                                     (path relative to GPT_SOVITS_DIR/EXP_ROOT/exp_name).
        inp_wav_dir (str): Input WAV directory path (absolute or relative to NOW_DIR).
                           Used in 1a and 1b.
        gpu_numbers (str): GPU indices, e.g., "0" or "0-1".
        bert_pretrained_dir (str): Pretrained BERT model path (relative to GPT_SOVITS_DIR).
        ssl_pretrained_dir (str): Pretrained SSL model path (relative to GPT_SOVITS_DIR).
        version (str): Model version ("v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4").
    """
    print(
        f"Running Step {step.upper()} for experiment: {exp_name}, version: {version}")
    inp_text = _clean_path(inp_text)
    inp_wav_dir = _clean_path(inp_wav_dir)
    gpu_list_for_iteration = _get_gpu_numbers_str_hyphenated(
        gpu_numbers).split('-')  # Split for iteration
    all_parts = len(gpu_list_for_iteration)

    opt_dir_script_pov = os.path.join(EXP_ROOT, exp_name)
    actual_opt_dir = os.path.join(GPT_SOVITS_DIR, opt_dir_script_pov)
    os.makedirs(actual_opt_dir, exist_ok=True)

    # Ensure necessary input files exist (check based on step)
    if step in {"1a", "1b"} and not os.path.exists(inp_wav_dir):
        raise FileNotFoundError(
            f"Input WAV directory not found: {inp_wav_dir}")
    # Check text file existence (path depends on step and is relative for 1b/1c)
    text_file_to_check = inp_text if step in ["1a", "1b", "1c"] else os.path.join(
        actual_opt_dir, "2-name2text.txt")
    if not os.path.exists(text_file_to_check):
        raise FileNotFoundError(
            f"Input text file not found: {text_file_to_check}")

    if step == "1a":
        print("--- Running Step 1A: Get Text ---")
        # Inp_text and inp_wav_dir are often absolute paths here
        # Pass them as is, let the script handle them or assume abs paths work
        # Webui logic passes them directly
        base_config = {
            "inp_text": inp_text,       
            "inp_wav_dir": inp_wav_dir,  
            "exp_name": exp_name,
            "opt_dir": opt_dir_script_pov,
            "bert_pretrained_dir": bert_pretrained_dir,
            "is_half": str(IS_HALF),
        }

        ps_list = []
        for i_part, gpu_id in enumerate(gpu_list_for_iteration):
            part_config = base_config.copy()
            part_config.update({
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": str(gpu_id.strip()),
            })
            cmd_list = [
                "-s", os.path.join("GPT_SoVITS", "prepare_datasets", "1-get-text.py")]
            # Use Popen directly to manage multiple processes for this step
            current_env = os.environ.copy()
            for k, v in part_config.items():
                current_env[k] = str(v)
            p = Popen([PYTHON_EXEC] + cmd_list, env=current_env, cwd=GPT_SOVITS_DIR,
                      stdout=PIPE, stderr=STDOUT, universal_newlines=True, bufsize=1)
            ps_list.append(p)

        # Wait for all processes and print their output
        for i, p in enumerate(ps_list):
            print(
                f"--- Output for 1A process {i+1}/{all_parts} (GPU {gpu_list_for_iteration[i]}) ---")
            for line in p.stdout:
                print(line.strip())
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(
                    f"Step 1A process {i+1} failed with return code {p.returncode}")
            print(f"--- Process {i+1} completed ---")

        # Merge output files
        merged_lines = []
        for i_part in range(all_parts):
            part_txt_path = os.path.join(
                actual_opt_dir, f"2-name2text-{i_part}.txt")
            if os.path.exists(part_txt_path):
                with open(part_txt_path, "r", encoding="utf8") as f:
                    merged_lines.extend(f.read().strip("\n").split("\n"))
                os.remove(part_txt_path)
            else:
                print(
                    f"Warning: Part file not found for 1a merge: {part_txt_path}")

        final_text_path = os.path.join(actual_opt_dir, "2-name2text.txt")
        with open(final_text_path, "w", encoding="utf8") as f:
            f.write("\n".join(line for line in merged_lines if line.strip()) + "\n")

        if not merged_lines:
            print("Warning: Step 1A produced no text.")
        print(f"Step 1A completed. Output: {final_text_path}")

    elif step == "1b":
        print("--- Running Step 1B: Get Hubert & Wav32k ---")
        # inp_text here should be path to 2-name2text.txt relative to GPT_SOVITS_DIR/EXP_ROOT/exp_name
        # processed_text_list_for_scripts = os.path.join(
        #     EXP_ROOT, exp_name, "2-name2text.txt")
        # inp_wav_dir is still the original wav dir path

        base_config = {
            "inp_text": inp_text,       
            "inp_wav_dir": inp_wav_dir,  
            "exp_name": exp_name,
            "opt_dir": opt_dir_script_pov,
            "cnhubert_base_dir": ssl_pretrained_dir,
            "sv_path": DEFAULT_SV_PATH,  # Use default SV path
            "is_half": str(IS_HALF),
        }

        ps_list = []
        for i_part, gpu_id in enumerate(gpu_list_for_iteration):
            part_config = base_config.copy()
            part_config.update({
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": str(gpu_id.strip()),
            })
            cmd_list = [
                "-s", os.path.join("GPT_SoVITS", "prepare_datasets", "2-get-hubert-wav32k.py")]
            current_env = os.environ.copy()
            for k, v in part_config.items():
                current_env[k] = str(v)
            p = Popen([PYTHON_EXEC] + cmd_list, env=current_env, cwd=GPT_SOVITS_DIR,
                      stdout=PIPE, stderr=STDOUT, universal_newlines=True, bufsize=1)
            ps_list.append(p)

        for i, p in enumerate(ps_list):
            print(
                f"--- Output for 1B Hub/Wav32k process {i+1}/{all_parts} (GPU {gpu_list_for_iteration[i]}) ---")
            for line in p.stdout:
                print(line.strip())
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(
                    f"Step 1B Hub/Wav32k process {i+1} failed with return code {p.returncode}")
            print(f"--- Process {i+1} completed ---")
        ps_list = []  # Reset list for SV step

        if version in PRO_SET:
            print(
                f"--- Running Step 1B additional SV extraction for Pro version: {version} ---")
            for i_part, gpu_id in enumerate(gpu_list_for_iteration):
                part_config = base_config.copy()  # sv_path is already in base_config
                part_config.update({
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": str(gpu_id.strip()),
                })
                cmd_list_sv = [
                    "-s", os.path.join("GPT_SoVITS", "prepare_datasets", "2-get-sv.py")]
                current_env = os.environ.copy()
                for k, v in part_config.items():
                    current_env[k] = str(v)
                p = Popen([PYTHON_EXEC] + cmd_list_sv, env=current_env, cwd=GPT_SOVITS_DIR,
                          stdout=PIPE, stderr=STDOUT, universal_newlines=True, bufsize=1)
                ps_list.append(p)

            for i, p in enumerate(ps_list):
                print(
                    f"--- Output for 1B SV process {i+1}/{all_parts} (GPU {gpu_list_for_iteration[i]}) ---")
                for line in p.stdout:
                    print(line.strip())
                p.wait()
                if p.returncode != 0:
                    raise RuntimeError(
                        f"Step 1B SV process {i+1} failed with return code {p.returncode}")
                print(f"--- Process {i+1} completed ---")

        print(f"Step 1B completed for experiment: {exp_name}")

    elif step == "1c":
        print("--- Running Step 1C: Get Semantic Tokens ---")
        # inp_text here should be path to 2-name2text.txt relative to GPT_SOVITS_DIR/EXP_ROOT/exp_name
        processed_text_list_for_scripts = os.path.join(
            EXP_ROOT, exp_name, "2-name2text.txt")

        pretrained_s2g_path = PRETRAINED_SOVITS_NAME_MAP.get(
            version)  # Get path from map
        if not pretrained_s2g_path or not os.path.exists(os.path.join(GPT_SOVITS_DIR, pretrained_s2g_path)):
            # Fallback to a default or raise error if no specific version pretrained model exists
            print(
                f"Warning: Pretrained SoVITS G model for version {version} not found at {pretrained_s2g_path}. Falling back to a default or check config.")
            # Use a known default if specific version is missing, or raise
            pretrained_s2g_path = PRETRAINED_SOVITS_NAME_MAP.get(
                "v2")  # Example fallback
            if not pretrained_s2g_path or not os.path.exists(os.path.join(GPT_SOVITS_DIR, pretrained_s2g_path)):
                raise FileNotFoundError(
                    f"Pretrained SoVITS G model for version {version} or fallback not found.")

        if version in PRO_SET:
            s2_config_filename = f"s2{version}.json"
        else:  # v1, v2, v3, v4 use s2.json for this step as per webui logic for 1Ac
            s2_config_filename = "s2.json"
        s2_config_file_path_script_pov = os.path.join(
            "GPT_SoVITS", "configs", s2_config_filename)
        actual_s2_config_path = os.path.join(
            GPT_SOVITS_DIR, s2_config_file_path_script_pov)
        if not os.path.exists(actual_s2_config_path):
            raise FileNotFoundError(
                f"SoVITS config file not found: {actual_s2_config_path}")

        base_config = {
            # Path relative to GPT_SOVITS_DIR/EXP_ROOT/exp_name
            "inp_text": inp_text,
            "exp_name": exp_name,
            "opt_dir": opt_dir_script_pov,
            "pretrained_s2G": pretrained_s2g_path,
            "s2config_path": s2_config_file_path_script_pov,
            "is_half": str(IS_HALF),
        }

        ps_list = []
        for i_part, gpu_id in enumerate(gpu_list_for_iteration):
            part_config = base_config.copy()
            part_config.update({
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": str(gpu_id.strip()),
            })
            cmd_list = [
                "-s", os.path.join("GPT_SoVITS", "prepare_datasets", "3-get-semantic.py")]
            current_env = os.environ.copy()
            for k, v in part_config.items():
                current_env[k] = str(v)
            p = Popen([PYTHON_EXEC] + cmd_list, env=current_env, cwd=GPT_SOVITS_DIR,
                      stdout=PIPE, stderr=STDOUT, universal_newlines=True, bufsize=1)
            ps_list.append(p)

        for i, p in enumerate(ps_list):
            print(
                f"--- Output for 1C process {i+1}/{all_parts} (GPU {gpu_list_for_iteration[i]}) ---")
            for line in p.stdout:
                print(line.strip())
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(
                    f"Step 1C process {i+1} failed with return code {p.returncode}")
            print(f"--- Process {i+1} completed ---")

        merged_lines = ["item_name\tsemantic_audio"]
        for i_part in range(all_parts):
            part_semantic_path = os.path.join(
                actual_opt_dir, f"6-name2semantic-{i_part}.tsv")
            if os.path.exists(part_semantic_path):
                with open(part_semantic_path, "r", encoding="utf8") as f:
                    lines = f.read().strip("\n").split("\n")
                    if lines and lines[0].lower().startswith("item_name"):
                        lines = lines[1:]
                    merged_lines.extend(lines)
                os.remove(part_semantic_path)
            else:
                print(
                    f"Warning: Part file not found for 1c merge: {part_semantic_path}")

        final_semantic_path = os.path.join(
            actual_opt_dir, "6-name2semantic.tsv")
        with open(final_semantic_path, "w", encoding="utf8") as f:
            f.write("\n".join(line for line in merged_lines if line.strip()) + "\n")

        print(f"Step 1C completed. Output: {final_semantic_path}")

    else:
        raise ValueError(
            f"Unsupported preprocessing step: {step}. Choose '1a', '1b', or '1c'.")


# --- Training Functions (1Ba, 1Bb) ---

def train_s1(exp_name: str, gpu_numbers: str, batch_size: int = None, total_epoch: int = 15,
             if_dpo: bool = False, if_save_latest: bool = True, if_save_every_weights: bool = True,
             save_every_epoch: int = 100, pretrained_s1: str = None, version: str = "v2Pro"):
    """
    Trains the GPT (S1) model. Corresponds to 1Bb in webui.

    Args:
        exp_name (str): Experiment name.
        gpu_numbers (str): GPU indices for DDP, e.g., "0" or "0,1". Passed to _CUDA_VISIBLE_DEVICES.
        batch_size (int, optional): Batch size per GPU. If None, script default is used (or calculate based on mem?).
        total_epoch (int, optional): Total training epochs. Defaults to 15.
        if_dpo (bool, optional): Whether to use DPO training. Defaults to False.
        if_save_latest (bool, optional): Whether to save only the latest checkpoint. Defaults to True.
        if_save_every_weights (bool, optional): Whether to save models every `save_every_epoch`. Defaults to True.
        save_every_epoch (int, optional): Epoch interval for saving models. Defaults to 100.
        pretrained_s1 (str, optional): Path to a pretrained S1 model (relative to GPT_SOVITS_DIR).
                                       If None, default based on `version` is used.
        version (str): Model version ("v1", "v2", etc.). Determines config and default pretrained model.
    """
    print(
        f"Starting GPT (S1) training for experiment: {exp_name}, version: {version}")
    cuda_visible_devices_str = _get_gpu_numbers_str_commaed(
        gpu_numbers)  # "0" or "0,1"

    # Determine pretrained S1 path
    s1_path = pretrained_s1 if pretrained_s1 else PRETRAINED_GPT_NAME_MAP.get(
        version)
    if not s1_path or not os.path.exists(os.path.join(GPT_SOVITS_DIR, s1_path)):
        print(
            f"Warning: Pretrained S1 model for version {version} not found at {s1_path}. Falling back to default v2 or check config.")
        s1_path = PRETRAINED_GPT_NAME_MAP.get("v2")  # Fallback example
        if not s1_path or not os.path.exists(os.path.join(GPT_SOVITS_DIR, s1_path)):
            raise FileNotFoundError(
                f"Pretrained S1 model (version {version} or fallback) not found at {s1_path}")

    # Determine S1 config file
    s1_config_filename = "s1longer.yaml" if version == "v1" else "s1longer-v2.yaml"
    s1_config_template_path_script_pov = os.path.join(
        "GPT_SoVITS", "configs", s1_config_filename)
    actual_s1_config_path = os.path.join(
        GPT_SOVITS_DIR, s1_config_template_path_script_pov)
    if not os.path.exists(actual_s1_config_path):
        raise FileNotFoundError(
            f"GPT S1 config file not found: {actual_s1_config_path}")

    with open(actual_s1_config_path, "r", encoding="utf8") as f:
        config_data = yaml.safe_load(f)

    s1_dir_script_pov = os.path.join(EXP_ROOT, exp_name)
    actual_s1_dir = os.path.join(GPT_SOVITS_DIR, s1_dir_script_pov)
    os.makedirs(os.path.join(actual_s1_dir, "logs_s1"), exist_ok=True)

    # Check required dataset files exist within the experiment directory
    required_semantic = os.path.join(actual_s1_dir, "6-name2semantic.tsv")
    required_phoneme = os.path.join(actual_s1_dir, "2-name2text.txt")
    if not os.path.exists(required_semantic) or not os.path.exists(required_phoneme):
        raise FileNotFoundError(
            f"Required dataset files not found in {actual_s1_dir}. Ensure steps 1a, 1b, 1c were run successfully.\nMissing: {required_semantic} or {required_phoneme}")

    current_batch_size = batch_size  # Use passed batch_size
    if current_batch_size is None:
        # Use a sensible default if not provided
        print("Batch size not specified for S1 training, using default 4.")
        current_batch_size = 4
        # webui has more complex batch size logic based on VRAM and is_half,
        # you might want to replicate that here if needed.
        if not IS_HALF:
            current_batch_size = max(
                1, current_batch_size // 2)  # Adjust for fp32

    if not IS_HALF:
        config_data["train"]["precision"] = "32"

    config_data["train"]["batch_size"] = current_batch_size
    config_data["train"]["epochs"] = total_epoch
    config_data["pretrained_s1"] = s1_path
    config_data["train"]["save_every_n_epoch"] = total_epoch
    config_data["train"]["if_save_every_weights"] = if_save_every_weights
    config_data["train"]["if_save_latest"] = if_save_latest
    config_data["train"]["if_dpo"] = if_dpo

    config_data["train"]["half_weights_save_dir"] = GPT_WEIGHT_ROOT_MAP.get(
        version, GPT_WEIGHT_ROOT_MAP.get("v2"))  # Default to v2 if version missing
    config_data["train"]["exp_name"] = exp_name
    # Paths for training data, relative to GPT_SOVITS_DIR
    config_data["train_semantic_path"] = os.path.join(
        s1_dir_script_pov, "6-name2semantic.tsv")
    config_data["train_phoneme_path"] = os.path.join(
        s1_dir_script_pov, "2-name2text.txt")
    # The actual logs dir name in script seems to be just logs_s1
    config_data["output_dir"] = os.path.join(s1_dir_script_pov, f"logs_s1")

    tmp_s1_config_path_script_pov = os.path.join("TEMP", "tmp_s1.yaml")
    actual_tmp_s1_config_path = os.path.join(
        GPT_SOVITS_TEMP_DIR, "tmp_s1.yaml")
    with open(actual_tmp_s1_config_path, "w", encoding="utf8") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    env_vars = {
        "_CUDA_VISIBLE_DEVICES": cuda_visible_devices_str,
        "hz": "25hz"  # As seen in webui.py
    }
    cmd_list = ["-s", os.path.join("GPT_SoVITS", "s1_train.py"),
                "--config_file", tmp_s1_config_path_script_pov]
    _run_subprocess(cmd_list, env_vars=env_vars, cwd=GPT_SOVITS_DIR)
    print(
        f"GPT (S1) training completed for experiment: {exp_name}, version: {version}")


def train_s2(exp_name: str, gpu_numbers: str, batch_size: int = None, total_epoch: int = 8,
             text_low_lr_rate: float = 0.4, if_save_latest: bool = True, if_save_every_weights: bool = True,
             save_every_epoch: int = 100, pretrained_s2G: str = None, pretrained_s2D: str = None,
             version: str = "v2Pro"):
    """
    Trains the SoVITS (S2) model. Corresponds to 1Ba in webui.

    Args:
        exp_name (str): Experiment name.
        gpu_numbers (str): GPU indices for DDP, e.g., "0" or "0-1". Passed to the config as 'gpu_numbers'.
        batch_size (int, optional): Batch size per GPU. If None, calculate based on IS_HALF or use default.
        total_epoch (int, optional): Total training epochs. Defaults to 8.
        text_low_lr_rate (float, optional): Text module learning rate weight (for non-v3/v4). Defaults to 0.4.
        if_save_latest (bool, optional): Whether to save only the latest checkpoint. Defaults to True.
        if_save_every_weights (bool, optional): Whether to save models every `save_every_epoch`. Defaults to True.
        save_every_epoch (int, optional): Epoch interval for saving models. Defaults to 100.
        pretrained_s2G (str, optional): Path to a pretrained S2G model (relative to GPT_SOVITS_DIR).
                                       If None, default based on `version` is used.
        pretrained_s2D (str, optional): Path to a pretrained S2D model (relative to GPT_SOVITS_DIR).
                                       If None, default based on `version` is used (derived from S2G if possible).
        version (str): Model version ("v1", "v2", etc.). Determines config, script, and default pretrained models.

    Additional version-specific args (not in original signature, but needed for v3/v4 compatibility):
        lora_rank (int, optional): LoRA rank (for v3/v4). Defaults to 32.
        if_grad_ckpt (bool, optional): Whether to enable gradient checkpointing (for v3/v4). Defaults to False.
        These are added directly below in the function signature.
    """
    # Add version-specific parameters that were missing in the original snippet but are in webui
    lora_rank: int = 32
    if_grad_ckpt: bool = False  # Only really applicable/implemented for v3 currently?

    print(
        f"Starting SoVITS (S2) training for experiment: {exp_name}, version: {version}")
    gpu_numbers_str_ddp = _get_gpu_numbers_str_hyphenated(
        gpu_numbers)  # "0" or "0-1"

    # Determine pretrained S2G and S2D paths
    s2g_path = pretrained_s2G if pretrained_s2G else PRETRAINED_SOVITS_NAME_MAP.get(
        version)
    s2d_path = pretrained_s2D if pretrained_s2D else get_s2d_path(
        s2g_path)  # Derive D if G is found and D not provided

    if not s2g_path or not os.path.exists(os.path.join(GPT_SOVITS_DIR, s2g_path)):
        print(
            f"Warning: Pretrained SoVITS G model for version {version} not found at {s2g_path}. Falling back to default v2 or check config.")
        s2g_path = PRETRAINED_SOVITS_NAME_MAP.get("v2")  # Fallback example
        if not s2g_path or not os.path.exists(os.path.join(GPT_SOVITS_DIR, s2g_path)):
            raise FileNotFoundError(
                f"Pretrained SoVITS G model (version {version} or fallback) not found at {s2g_path}")
        s2d_path = get_s2d_path(s2g_path)  # Update D path based on fallback G

    # Check D path exists (it might be optional depending on script/version)
    if s2d_path and not os.path.exists(os.path.join(GPT_SOVITS_DIR, s2d_path)):
        print(
            f"Warning: Pretrained SoVITS D model not found at {s2d_path}. Training might proceed without it or fail.")

    # Determine config file and script based on version
    if version in V3_V4_SET:
        train_script_name = "s2_train_v3_lora.py"
        # v3/v4 LoRA training uses a base s2.json and modifies for LoRA
        s2_config_filename = "s2.json"
    elif version in PRO_SET:
        s2_config_filename = f"s2{version}.json"
        train_script_name = "s2_train.py"
    else:  # v1, v2
        s2_config_filename = "s2.json"
        train_script_name = "s2_train.py"

    s2_config_template_path_script_pov = os.path.join(
        "GPT_SoVITS", "configs", s2_config_filename)
    actual_s2_config_path = os.path.join(
        GPT_SOVITS_DIR, s2_config_template_path_script_pov)
    if not os.path.exists(actual_s2_config_path):
        raise FileNotFoundError(
            f"SoVITS S2 config file not found: {actual_s2_config_path}")

    with open(actual_s2_config_path, "r", encoding="utf8") as f:
        config_data = json.load(f)

    s2_dir_script_pov = os.path.join(EXP_ROOT, exp_name)
    actual_s2_dir = os.path.join(GPT_SOVITS_DIR, s2_dir_script_pov)
    os.makedirs(os.path.join(actual_s2_dir,
                f"logs_s2_{version}"), exist_ok=True)

    # Check required dataset files exist within the experiment directory
    # S2 uses hubert files and semantic files.
    required_semantic = os.path.join(actual_s2_dir, "6-name2semantic.tsv")
    if not os.path.exists(required_semantic):
        raise FileNotFoundError(
            f"Required dataset files for S2 not found in {actual_s2_dir}. Ensure steps 1b and 1c were run successfully.\nMissing: {required_semantic}")

    current_batch_size = batch_size  # Use passed batch_size
    if current_batch_size is None:
        # Use a sensible default if not provided
        print("Batch size not specified for S2 training, using default 4.")
        current_batch_size = 4
        # webui has more complex batch size logic based on VRAM and is_half,
        # you might want to replicate that here if needed.
        if not IS_HALF:  # Often batch size is halved for fp32
            current_batch_size = max(1, current_batch_size // 2)

    if not IS_HALF:
        config_data["train"]["fp16_run"] = False

    config_data["train"]["batch_size"] = current_batch_size
    config_data["train"]["epochs"] = total_epoch
    config_data["train"]["pretrained_s2G"] = s2g_path
    config_data["train"]["pretrained_s2D"] = s2d_path
    config_data["train"]["if_save_latest"] = if_save_latest
    config_data["train"]["if_save_every_weights"] = if_save_every_weights
    config_data["train"]["save_every_epoch"] = total_epoch
    # DDP script handles parsing this
    config_data["train"]["gpu_numbers"] = gpu_numbers_str_ddp
    # Important for model architecture selection within script
    config_data["model"]["version"] = version
    # Relative to GPT_SOVITS_DIR
    config_data["data"]["exp_dir"] = config_data["s2_ckpt_dir"] = s2_dir_script_pov

    config_data["save_weight_dir"] = SOVITS_WEIGHT_ROOT_MAP.get(
        version, SOVITS_WEIGHT_ROOT_MAP.get("v2"))  # Default to v2 if version missing
    config_data["name"] = exp_name
    # config_data["version"] = version # This is often used for logging/output, model.version is for architecture

    # Version-specific training params
    if version in V3_V4_SET:
        config_data["train"]["lora_rank"] = lora_rank
        config_data["train"]["grad_ckpt"] = if_grad_ckpt
        # Remove text_low_lr_rate if present, as it's not for LoRA
        if "text_low_lr_rate" in config_data["train"]:
            del config_data["train"]["text_low_lr_rate"]
    else:
        config_data["train"]["text_low_lr_rate"] = text_low_lr_rate
        # Remove lora_rank and grad_ckpt if present
        if "lora_rank" in config_data["train"]:
            del config_data["train"]["lora_rank"]
        if "grad_ckpt" in config_data["train"]:
            # grad_ckpt might not be applicable or handled differently
            del config_data["train"]["grad_ckpt"]

    tmp_s2_config_path_script_pov = os.path.join("TEMP", "tmp_s2.json")
    actual_tmp_s2_config_path = os.path.join(
        GPT_SOVITS_TEMP_DIR, "tmp_s2.json")
    with open(actual_tmp_s2_config_path, "w", encoding="utf8") as f:
        json.dump(config_data, f, indent=2)

    cmd_list = ["-s", os.path.join("GPT_SoVITS", train_script_name),
                "--config", tmp_s2_config_path_script_pov]
    _run_subprocess(cmd_list, cwd=GPT_SOVITS_DIR)
    print(
        f"SoVITS (S2) training completed for experiment: {exp_name}, version: {version}")


# --- Example Usage ---
if __name__ == "__main__":
    # Example usage:
    # Ensure your dataset is prepared and paths are correct.
    # Paths for inp_text and inp_wav_dir should be absolute or resolvable.

    EXP_NAME = "my_char_v2pro_test"  # Change experiment name as needed
    # CHANGE THIS TO TEST DIFFERENT VERSIONS (v1, v2, v2Pro, v2ProPlus, v3, v4)
    TARGET_VERSION = "v2Pro"

    # --- Dataset Path Configuration ---
    # These paths point to your raw/processed dataset files BEFORE formatting steps 1a/b/c.
    # They should be accessible by communication.py (absolute paths recommended).
    # If you have used the 0a, 0b, 0c steps from webui, these paths will point
    # to the output of those steps (e.g., output/slicer_opt, output/asr_opt/xxx.list)

    # !!! IMPORTANT !!!
    # Replace with your actual dataset paths after running slicing/ASR/labeling tools
    # For example, if you sliced audio to D:\MyDataset\sliced_wavs
    # and ran ASR + Labeling on it resulting in D:\MyDataset\labels\my_char.list
    # INP_TEXT_LIST_PATH_INITIAL = r"D:\MyDataset\labels\my_char.list"
    # INP_WAV_DIR_INITIAL = r"D:\MyDataset\sliced_wavs"
    #
    # For this example, we'll create dummy data if it doesn't exist.
    # This assumes your project structure looks something like:
    # YourProject/
    # ├── communication.py
    # ├── thirdparty/
    # │   └── GPTSoVITS/
    # └── dataset_example/ # <-- This is where the dummy data will be created
    #     └── my_char_v2pro_test/
    #         ├── my_char_v2pro_test.list
    #         └── wavs/

    # Adjust path relative to where you run this script
    YOUR_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(
        __file__), ".."))  # Assuming script is in YourProject/somefolder/
    # If script is directly in YourProject/, use: YOUR_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

    BASE_DATA_PATH = os.path.join(
        YOUR_PROJECT_ROOT, "dataset_example", EXP_NAME)
    os.makedirs(BASE_DATA_PATH, exist_ok=True)

    DUMMY_LIST_FILE = os.path.join(BASE_DATA_PATH, f"{EXP_NAME}.list")
    DUMMY_WAV_DIR = os.path.join(BASE_DATA_PATH, "wavs")
    os.makedirs(DUMMY_WAV_DIR, exist_ok=True)

    # Create dummy input files for testing if they don't exist
    if not os.path.exists(DUMMY_LIST_FILE) or not os.listdir(DUMMY_WAV_DIR):
        print("Creating dummy dataset...")
        try:
            from scipy.io.wavfile import write as write_wav
            import numpy as np
            samplerate = 32000  # Target for Hubert
            if not os.path.exists(DUMMY_LIST_FILE):
                with open(DUMMY_LIST_FILE, "w", encoding="utf-8") as f:
                    f.write(f"dummy1.wav|{EXP_NAME}|ZH|你好世界。\n")
                    f.write(f"dummy2.wav|{EXP_NAME}|ZH|这是一个测试。\n")
                print(f"Created dummy list file: {DUMMY_LIST_FILE}")
            if not os.listdir(DUMMY_WAV_DIR):
                for i in [1, 2]:
                    duration = 3  # seconds
                    frequency = 440  # Hz
                    t = np.linspace(0, duration, int(
                        samplerate * duration), endpoint=False)
                    amplitude = np.iinfo(np.int16).max * 0.5
                    audio_data = amplitude * np.sin(2 * np.pi * frequency * t)
                    write_wav(os.path.join(
                        DUMMY_WAV_DIR, f"dummy{i}.wav"), samplerate, audio_data.astype(np.int16))
                print(f"Created dummy wav files in: {DUMMY_WAV_DIR}")
        except ImportError:
            print("Warning: scipy or numpy not found. Cannot create dummy dataset. Please install them (`pip install scipy numpy`) or provide your own dataset.")
            print("Skipping dataset creation.")

    # Path to the initial list file (output of 0d labeling)
    INP_TEXT_LIST_PATH_INITIAL = DUMMY_LIST_FILE
    # Directory containing wav files named in the .list (output of 0b slicing)
    INP_WAV_DIR_INITIAL = DUMMY_WAV_DIR

    # --- GPU Configuration ---
    # Specify GPUs as a string, e.g., "0", "1", "0-1", "0,1"
    GPUS_TO_USE_STR = "0"  # Change this to match your available GPUs

    print(
        f"\n=== Starting training pipeline for version {TARGET_VERSION}, experiment '{EXP_NAME}' ===")
    print(f"Initial Text List: {INP_TEXT_LIST_PATH_INITIAL}")
    print(f"Initial WAV Directory: {INP_WAV_DIR_INITIAL}")
    print(f"GPUs configured: {GPUS_TO_USE_STR}")
    print(f"Using Python executable: {PYTHON_EXEC}")
    print(f"Using half precision (IS_HALF): {IS_HALF}")
    print(f"GPT-SoVITS directory: {GPT_SOVITS_DIR}")

    # Path to the 2-name2text.txt generated by step 1a (relative to GPT_SOVITS_DIR/EXP_ROOT/exp_name)
    # This path is used by 1b and 1c scripts internally via config/env vars.
    PROCESSED_TEXT_LIST_SCRIPT_POV = os.path.join(
        EXP_ROOT, EXP_NAME, "2-name2text.txt")

    try:
        # 1. Dataset Formatting
        # These steps will create/update files in thirdparty/GPTSoVITS/logs/EXP_NAME/

        # Step 1A: Get Text (Tokenization, Phonemization)
        # This step reads the initial list and wav dir.
        preprocess_dataset(
            step="1a",
            exp_name=EXP_NAME,
            inp_text=INP_TEXT_LIST_PATH_INITIAL,  # Path to the original list file
            inp_wav_dir=INP_WAV_DIR_INITIAL,     # Path to the original sliced wav directory
            gpu_numbers=GPUS_TO_USE_STR,
            bert_pretrained_dir=DEFAULT_BERT_PRETRAINED_DIR,  # Use default or specify
            ssl_pretrained_dir=DEFAULT_SSL_PRETRAINED_DIR,  # Use default or specify
            version=TARGET_VERSION  # Pass version
        )

        # Step 1B: Get Hubert & Wav32k (SSL Features)
        # This step reads the 2-name2text.txt and the original wav dir.
        preprocess_dataset(
            step="1b",
            exp_name=EXP_NAME,
            # Pass the path relative to the script's CWD
            inp_text=PROCESSED_TEXT_LIST_SCRIPT_POV,
            inp_wav_dir=INP_WAV_DIR_INITIAL,         # Still the original wav dir path
            gpu_numbers=GPUS_TO_USE_STR,
            # Not used in 1b script but kept for signature
            bert_pretrained_dir=DEFAULT_BERT_PRETRAINED_DIR,
            ssl_pretrained_dir=DEFAULT_SSL_PRETRAINED_DIR,  # Use default or specify
            version=TARGET_VERSION  # Pass version
        )

        # Step 1C: Get Semantic Tokens
        # This step reads the 2-name2text.txt and uses the pretrained SoVITS G model.
        # It does NOT need inp_wav_dir directly.
        preprocess_dataset(
            step="1c",
            exp_name=EXP_NAME,
            inp_text=PROCESSED_TEXT_LIST_SCRIPT_POV,  # Path relative to the script's CWD
            inp_wav_dir="",  # Not used in 1c script
            gpu_numbers=GPUS_TO_USE_STR,
            bert_pretrained_dir=DEFAULT_BERT_PRETRAINED_DIR,  # Not used in 1c script
            ssl_pretrained_dir=DEFAULT_SSL_PRETRAINED_DIR,  # Not used in 1c script
            version=TARGET_VERSION  # Pass version
        )

        # 2. Training
        # Training will use the prepared dataset files in thirdparty/GPTSoVITS/logs/EXP_NAME/

        # SoVITS Training (S2)
        # Defaults for batch size, epochs, etc. can be customized here
        # based on version or hardware, replicating webui's set_default logic.
        sovits_batch_size = 4
        sovits_total_epoch = 8
        sovits_save_every = 4
        if TARGET_VERSION in V3_V4_SET:
            sovits_batch_size = 2
            sovits_total_epoch = 4  # Shorter for LoRA
            sovits_save_every = 2
            sovits_lora_rank = 32
            sovits_grad_ckpt = True
        else:
            sovits_lora_rank = 32  # Still pass default even if not used by script
            sovits_grad_ckpt = False
            sovits_text_low_lr_rate = 0.4  # Use for non-v3/v4

        print("\n--- Running SoVITS (S2) Training ---")
        train_s2(
            exp_name=EXP_NAME,
            gpu_numbers=GPUS_TO_USE_STR,  # Pass the GPU string
            batch_size=sovits_batch_size,
            total_epoch=sovits_total_epoch,
            save_every_epoch=sovits_save_every,
            # Pass all potential version-specific params, the function decides which to use
            # Use default if not applicable
            text_low_lr_rate=sovits_text_low_lr_rate if TARGET_VERSION not in V3_V4_SET else 0.4,
            # lora_rank and if_grad_ckpt are passed directly in the function call now
            # since they were added to the signature (conceptually, if not in original snippet)
            # This requires slightly modifying the original train_s2 signature defined above,
            # or passing them as kwargs and modifying the function to accept kwargs.
            # Let's add them to the function signature for clarity.
            # (Update: Added them directly to the train_s2 definition)
            if_save_latest=True,
            if_save_every_weights=True,
            # pretrained_s2G and pretrained_s2D are handled internally based on version if None
            pretrained_s2G=None,
            pretrained_s2D=None,
            version=TARGET_VERSION,
            # Pass the added version-specific params
            lora_rank=sovits_lora_rank,
            if_grad_ckpt=sovits_grad_ckpt
        )

        # GPT Training (S1)
        gpt_batch_size = 4
        gpt_total_epoch = 15
        gpt_save_every = 5

        print("\n--- Running GPT (S1) Training ---")
        train_s1(
            exp_name=EXP_NAME,
            gpu_numbers=GPUS_TO_USE_STR,  # Pass the GPU string
            batch_size=gpt_batch_size,
            total_epoch=gpt_total_epoch,
            save_every_epoch=gpt_save_every,
            if_dpo=False,
            if_save_latest=True,
            if_save_every_weights=True,
            # pretrained_s1 is handled internally based on version if None
            pretrained_s1=None,
            version=TARGET_VERSION
        )

        print(
            f"\n\n=== All training steps for version {TARGET_VERSION}, experiment '{EXP_NAME}' completed successfully! ===")

    except FileNotFoundError as e:
        print(f"\n\nXXX Required file or directory not found: {e} XXX")
        print("Please ensure dataset paths are correct and previous preprocessing steps completed successfully.")
        # traceback.print_exc() # Uncomment for full traceback

    except RuntimeError as e:
        print(f"\n\nXXX A subprocess failed: {e} XXX")
        print("Check the output above for specific error messages from the subprocess.")
        # traceback.print_exc() # Uncomment for full traceback

    except Exception as e:
        print(f"\n\nXXX An unexpected error occurred: {e} XXX")
        traceback.print_exc()  # Print traceback for unexpected errors
