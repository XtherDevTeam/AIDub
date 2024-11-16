# Custom Data Providers

This folder contains the custom data providers for AIDub.

Each module in this folder should implement the following methods:

- `fetch_vo_urls(url: str, target_va: str)` - This method should fetch the voice collections from the given URL and return it as a string.
    For example, checkout `huggingface.py` which fetches the voice collections from Hugging Face.