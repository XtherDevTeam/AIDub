"""
https://github.com/XtherDevTeam/AIDub

AIDub Middleware API client for Python.

Copyright (C) 2024 Jerry Chou, All rights reserved.
Open-source under the MIT license.
"""


import requests
import typing

class AIDubAPIError(Exception):
    pass

class AIDubMiddlewareAPI():
    def __init__(self, url):
        self.url = url
        
    def data_if_ok_else_raise_error(self, resp_json) -> dict[typing.Any, typing.Any]:
        if resp_json["status"]:
            return resp_json["data"]
        else:
            raise AIDubAPIError(resp_json["data"])
        
        
    def info(self):
        try:
            response = requests.post(self.url + "/info")
            return self.data_if_ok_else_raise_error(response.json())
        except requests.exceptions.RequestException as e:
            raise AIDubAPIError(e)
        
        
    def download_dataset(self, char_names: list[str], sources_to_fetch: list[str]):
        try:
            response = requests.post(self.url + "/download_dataset", json={"char_names": char_names, "sources_to_fetch": sources_to_fetch})
            return self.data_if_ok_else_raise_error(response.json())
        except requests.exceptions.RequestException as e:
            raise AIDubAPIError(e)
        
        
    def emotion_classification(self):
        try:
            response = requests.post(self.url + "/emotion_classification")
            return self.data_if_ok_else_raise_error(response.json())
        except requests.exceptions.RequestException as e:
            raise AIDubAPIError(e)
        
        
    def dub(self, text: str, char_name: str):
        try:
            response = requests.post(self.url + "/dub", json={"text": text, "char_name": char_name})
            return self.data_if_ok_else_raise_error(response.json())
        except requests.exceptions.RequestException as e:
            raise AIDubAPIError(e)
        
    
    def data_preprocessing_get_text(self):
        try:
            response = requests.post(self.url + "/gpt_sovits/dataset_preprocessing/get_text")
            return self.data_if_ok_else_raise_error(response.json())
        except requests.exceptions.RequestException as e:
            raise AIDubAPIError(e)
        
    
    def data_preprocessing_get_hubert_wav32k(self):
        try:
            response = requests.post(self.url + "/gpt_sovits/dataset_preprocessing/get_hubert_wav32k")
            return self.data_if_ok_else_raise_error(response.json())
        except requests.exceptions.RequestException as e:
            raise AIDubAPIError(e)
        
        
    # dataset_preprocessing/name_to_semantic
    def data_preprocessing_name_to_semantic(self):
        try:
            response = requests.post(self.url + "/gpt_sovits/dataset_preprocessing/name_to_semantic")
            return self.data_if_ok_else_raise_error(response.json())
        except requests.exceptions.RequestException as e:
            raise AIDubAPIError(e)
        
        
    # train_model_gpt
    def train_model_gpt(self, batch_size: int = None, total_epoch: int = 15):
        try:
            response = requests.post(self.url + "/gpt_sovits/train_model_gpt", json={"batch_size": batch_size, "total_epoch": total_epoch})
            return self.data_if_ok_else_raise_error(response.json())
        except requests.exceptions.RequestException as e:
            raise AIDubAPIError(e)
        
    
    # train_model_sovits
    def train_model_sovits(self, batch_size: int = None, total_epoch: int = 15):
        try:
            response = requests.post(self.url + "/gpt_sovits/train_model_sovits", json={"batch_size": batch_size, "total_epoch": total_epoch})
            return self.data_if_ok_else_raise_error(response.json())
        except requests.exceptions.RequestException as e:
            raise AIDubAPIError(e)