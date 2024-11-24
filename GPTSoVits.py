import common
import requests
import urllib.parse
import subprocess


class GPTSoVitsAPI():
    """
    GPTSoVits API class.

    Attributes:
        api_url (str): Url of the GPTSoVits API.
        usingV3 (bool): whether using APIv3.py from the fast_inference_ branch.
        ttsInferYamlPath (str): TTS inference YAML path. Necessary for v3 API.

    Methods:
        tts_v1(ref_audio: str, ref_text: str, text: str, ref_language: str = 'auto', text_language: str = 'auto') -> requests.Response: Run TTS inference using the classic v1 API.
        tts_v3(ref_audio: str, ref_text: str, text: str, ref_language: str = 'auto', text_language: str = 'auto') -> requests.Response: Run TTS inference using the v3 API from the fast_inference_ branch.
        tts(ref_audio: str, ref_text: str, text: str, ref_language: str = 'auto', text_language: str = 'auto') -> requests.Response: Run TTS inference based on the API version.
        changeReferenceAudio(ref_audio: str, ref_text: str, ref_language: str = 'auto') -> None: Change reference audio.
        control(command: str): Restart or exit.
    """

    def __init__(self, api_url: str, isTTSv3: bool = False, ckpt_path: str = None, pth_path: str = None) -> None:
        """
        Initialize the GPTSoVitsAPI.

        Args:
            api_url (str): Url of the GPTSoVits API.
            isTTSv3 (bool): whether using APIv3.py from the fast_inference_ branch. Defaults to False.
            ttsInferYamlPath (str): TTS inference YAML path. Necessary for v3 API.
        """
        self.api_url = api_url
        self.usingV3 = isTTSv3
        if isTTSv3:
            common.log('Using v3 API from the fast_inference_ branch')
            self.set_models(ckpt_path, pth_path)
        else:
            common.log('Using classic v1 API from the main branch')


    def set_models(self, ckpt_path: str, pth_path: str) -> None:
        """
        Set the models for the v3 API.

        Args:
            ckpt_path (str): Path to the TTS model checkpoint.
            pth_path (str): Path to the TTS model state dict.
        """
        a = requests.get(f'{self.api_url}/set_gpt_weights', params={
            'weights_path': ckpt_path
        })
        b = requests.get(f'{self.api_url}/set_sovits_weights', params={
            'weights_path': pth_path
        })
        if a.status_code == 200 and b.status_code == 200:
            common.log('Successfully set models for v3 API')
        else:
            common.panic(f'Failed to set models for v3 API: {a.content} {b.content}')


    # text to speech function for v1 API
    def tts_v1(self, ref_audio: str, ref_text: str, text: str, ref_language: str = 'auto', text_language: str = 'auto') -> requests.Response:
        """
        Run TTS inference using the classic v1 API.

        Args:
            ref_audio (str): Reference audio path.
            ref_text (str): Reference text.
            text (str): Text to be synthesized.
            ref_language (str, optional): Reference audio language. Defaults to 'auto'.
            text_language (str, optional): Text language. Defaults to 'auto'.

        Returns:
            requests.Response: Response object from the API.
        """
        return requests.post(f'{self.api_url}/', json={
            "refer_wav_path": ref_audio,
            "prompt_text": ref_text,
            "prompt_language": ref_language,
            "text": text,
            "text_language": text_language
        }, stream=True)


    def build_tts_v3_request(self, ref_audio: str, ref_text: str, text: str, ref_language: str = 'auto', text_language: str = 'auto') -> str:
        """
        Build the request for the v3 API.

        Args:
            ref_audio (str): Reference audio path.
            ref_text (str): Reference text.
            text (str): Text to be synthesized.
            ref_language (str, optional): Reference audio language. Defaults to 'auto'.
            text_language (str, optional): Text language. Defaults to 'auto'.

        Returns:
            str: Request string for the v3 API.
        """
        return f'''{self.api_url}/tts?{urllib.parse.urlencode({
             "text": text,
            "text_lang": text_language,
            "ref_audio_path": ref_audio,
            "prompt_text": ref_text,
            "prompt_lang": ref_language,
            "media_type": "aac",
            "streaming_mode": True,
            "parallel_infer": False,
            "tts_infer_yaml_path": self.ttsInferYamlPath
        })}'''


    def tts_v3(self, ref_audio: str, ref_text: str, text: str, ref_language: str = 'auto', text_language: str = 'auto', streamed: bool = False) -> requests.Response:
        """
        Run TTS inference using the v3 API from the fast_inference_ branch.

        Args:
            ref_audio (str): Reference audio path. Should be less than 10 seconds long.
            ref_text (str): Reference text.
            text (str): Text to be synthesized.
            ref_language (str, optional): Reference audio language. Defaults to 'auto'.
            text_language (str, optional): Text language. Defaults to 'auto'.

        Returns:
            requests.Response: Response object from the API.
        """
        return requests.get(f'{self.api_url}/tts', params={
            "text": text,
            "text_lang": text_language,
            "ref_audio_path": ref_audio,
            "prompt_text": ref_text,
            "prompt_lang": ref_language,
            "media_type": "aac",
            "streaming_mode": True,
            "parallel_infer": False
        }, stream=streamed)

    def tts(self, ref_audio: str, ref_text: str, text: str, ref_language: str = 'auto', text_language: str = 'auto', streamed: bool = False) -> requests.Response:
        """
        Run TTS inference based on the API version.

        Args:
            ref_audio (str): Reference audio path.
            ref_text (str): Reference text.
            text (str): Text to be synthesized.
            ref_language (str, optional): Reference audio language. Defaults to 'auto'.
            text_language (str, optional): Text language. Defaults to 'auto'.
            streamed (bool, optional): Whether to stream the response. Makes no difference for v1 API. Defaults to False.

        Returns:
            requests.Response: Response object from the API.
        """
        if self.usingV3:
            return self.tts_v3(ref_audio, ref_text, text, ref_language, text_language, streamed)
        else:
            return self.tts_v1(ref_audio, ref_text, text, ref_language, text_language)

    # change reference audio
    def changeReferenceAudio(self, ref_audio: str, ref_text: str, ref_language: str = 'auto') -> None:
        r = requests.post(f'{self.api_url}/change_ref', json={
            "refer_wav_path": ref_audio,
            "prompt_text": ref_text,
            "prompt_language": ref_language
        })
        if r.status_code == 400:
            raise RuntimeError(f'{__name__}: Failed to change reference audio')
        else:
            return

    # restart or exit
    def control(self, command: str):
        requests.post(f'{self.api_url}/control', json={
            "command": command})

def run_get_text() -> subprocess.Popen:
    """
    Run the get_text.py script to get text from the user.

    Returns:
        subprocess.Popen: Popen object of the get_text.py script.
    """
    current_python_exec = subprocess.check_output(['which', 'python3']).decode().strip()
    
    subprocess.Popen()