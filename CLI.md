# AIDub Command-Line Interface (CLI) Tool

## Introduction

This document describes the AIDub Command-Line Interface (CLI) tool, which allows you to directly access AIDub functionalities from your terminal without needing to use the HTTP API. This tool provides a convenient way to manage datasets, perform emotion classification, dub text, preprocess datasets, and train models directly from the command line.

## Prerequisites

*   **Python:** Ensure you have Python installed on your system.
*   **AIDub Environment:**  You should have the AIDub codebase set up and configured as described in the original documentation. This CLI tool is designed to be run within the AIDub environment, assuming necessary dependencies and configurations are in place.
*   **Save the Script:** Save the provided Python script (e.g., as `cli.py`) in your AIDub project directory or a location where Python can access the AIDub modules (ensure your `PYTHONPATH` is correctly set if running from outside the AIDub project root).

## Usage

```bash
python cli.py <command> <options>
```

Replace `<command>` with one of the available commands listed below, and `<options>` with the specific arguments for that command.

## Available Commands

### 1. `download_dataset`

**Description:** Downloads a dataset for specified characters from configured data sources.

**Arguments:**

| Argument        | Description                                  | Required | Type   | Example                                   |
| --------------- | -------------------------------------------- | -------- | ------ | ----------------------------------------- |
| `--char-names`    | Comma-separated list of character names.     | Yes      | String | `"character1,character2"`                |
| `--sources-to-fetch` | Comma-separated list of data sources to fetch from. | Yes      | String | `"source1,source2"`                       |

**Example Usage:**

```bash
python cli.py download_dataset --char-names "Alice,Bob" --sources-to-fetch "fandom_source,custom:myprovider:https://example.com"
```

This command will download datasets for characters "Alice" and "Bob" from the specified sources.

### 2. `emotion_classification`

**Description:** Performs emotion classification on the existing dataset.

**Arguments:**

This command has no arguments.

**Example Usage:**

```bash
python cli.py emotion_classification
```

This command will initiate the emotion classification process.

### 3. `dub`

**Description:** Dubs the provided text using the specified character's voice model.

**Arguments:**

| Argument    | Description                   | Required | Type   | Example                         |
| ----------- | ----------------------------- | -------- | ------ | ------------------------------- |
| `--text`      | The text to be dubbed.        | Yes      | String | `"Hello, how are you today?"`    |
| `--char-name` | The name of the character to use for dubbing. | Yes      | String | `"CharacterName"`                 |

**Example Usage:**

```bash
python cli.py dub --text "This is a test dubbing." --char-name "CharacterA"
```

This command will dub the text "This is a test dubbing." using the voice model of "CharacterA" and save the output audio file as `CharacterA.aac` in the current directory.

### 4. `preprocess_dataset`

This command group includes subcommands for different dataset preprocessing steps.

#### 4.1. `preprocess_dataset get_text`

**Description:** Preprocesses the dataset to generate text lists for each character.

**Arguments:**

This command has no arguments.

**Example Usage:**

```bash
python cli.py preprocess_dataset get_text
```

#### 4.2. `preprocess_dataset get_hubert_wav32k`

**Description:** Preprocesses the dataset to generate Hubert features (wav32k) for each character. Skips characters if models already exist (assuming preprocessing is already done).

**Arguments:**

This command has no arguments.

**Example Usage:**

```bash
python cli.py preprocess_dataset get_hubert_wav32k
```

#### 4.3. `preprocess_dataset name_to_semantic`

**Description:** Preprocesses the dataset to generate semantic embeddings for each character. Skips characters if models already exist (assuming preprocessing is already done).

**Arguments:**

This command has no arguments.

**Example Usage:**

```bash
python cli.py preprocess_dataset name_to_semantic
```

### 5. `train_model`

This command group includes subcommands for training different models.

#### 5.1. `train_model gpt`

**Description:** Trains the GPT (stage 1) model for specified characters. Skips characters if GPT models already exist.

**Arguments:**

| Argument      | Description                                | Required | Type   | Default