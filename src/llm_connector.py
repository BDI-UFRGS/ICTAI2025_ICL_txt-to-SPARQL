from enum import Enum
from groq import Groq
import requests
import os
import google.generativeai as genai
from huggingface_hub import InferenceClient
import json
import time

class SupportedModels(Enum):
    LLAMA3_70B_8192 = 'llama3-70b-8192'
    GEMINI_1_5_FLASH = 'gemini-1.5-flash'
    MISTRAL_7B_INSTRUCT = 'mistral-7b-instruct-v0.1'
    HF_LLAMA3_70B_INSTRUCT = 'meta-llama/Llama-3.3-70B-Instruct'
    HF_DEEP_SEEK_R1 = "deepseek-ai/DeepSeek-R1"

class LLMApi:

    def __init__(self, model: SupportedModels):
        self._model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: SupportedModels):
        self._model = model
    
    def get_model_response(self, input_text: str):
        if self._model == SupportedModels.LLAMA3_70B_8192:
            time.sleep(7)
            return GroqApiConnector(model=self.model, max_tokens=8192).get_model_response(input_text)
        if self.model == SupportedModels.GEMINI_1_5_FLASH:
            return GoogleAPIConnector(model=self.model).get_model_response(input_text)
        if self.model == SupportedModels.MISTRAL_7B_INSTRUCT:
            return LMStudioApiConnector(model=self.model).get_model_response(input_text)
        if self.model == SupportedModels.HF_LLAMA3_70B_INSTRUCT or self.model == SupportedModels.HF_DEEP_SEEK_R1:
            return HFApiConnector(model=self.model).get_model_response(input_text)

class GroqApiConnector:
    
    def __init__(self, model: SupportedModels, api_key: str = os.getenv('GROQ_API_KEY'), sleep_time: int = 0, temperature: float = 1.0, max_tokens: int = 1024):
        self._api_key = api_key
        self._model = model
        self._sleep_time = sleep_time
        self._temperature = temperature
        self._api_key = api_key
        self.client = Groq(api_key=self._api_key)
        self._max_tokens = max_tokens

    @property
    def api_key(self):
        return self._api_key
    
    @property
    def model(self):
        return self._model
    
    @property
    def sleep_time(self):
        return self._sleep_time
    
    @property
    def temperature(self):
        return self._temperature
    
    @property
    def max_tokens(self):
        return self._max_tokens
    
    @api_key.setter
    def api_key(self, api_key: str):
        self._api_key = api_key

    @model.setter
    def model(self, model: str):
        self._model = model

    @sleep_time.setter
    def sleep_time(self, sleep_time: int):
        self._sleep_time = sleep_time

    @temperature.setter
    def temperature(self, temperature: float):
        self._temperature = temperature

    @max_tokens.setter
    def max_tokens(self, max_tokens: int):
        self._max_tokens = max_tokens

    def get_model_response(self, input_text: str):
        completion = self.client.chat.completions.create(
                model=self.model.value,
                messages=[
                    {
                        "role": "user",
                        "content": input_text
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                stream=False,
                stop=None,
        )

        return completion.choices[0].message.content

class GoogleAPIConnector:

    def __init__(self, model: SupportedModels, api_key: str = os.getenv('GROQ_API_KEY')):
        self._model = genai.GenerativeModel(model.value)
        self._api_key = api_key

    def get_model_response(self, input_text: str):
        genai.configure(api_key=self._api_key)
        return self._model.generate_content(input_text).text

class LMStudioApiConnector:

    def __init__(self, model: SupportedModels, url: str = "http://172.28.160.1:1234/v1/chat/completions"):
        self._model = model
        self._url = url
        self._headers = {"Content-Type": "application/json",}

    @property
    def model(self):
        return self._model

    @property
    def url(self):
        return self._url

    @property
    def headers(self):
        return self._headers
    
    def get_model_response(self, input_text: str):
        data = {
            "model": self.model.value,
            "messages": [
                {"role": "system", "content": input_text}
            ],
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False
        }

        response = requests.post(self.url, headers=self.headers, json=data)

        return response.json().get("choices", [{}])[0].get("message", {}).get("content")
    
class HFApiConnector:
    
    def __init__(self, model: SupportedModels, token : str = os.getenv('HF_TOKEN'), timeout : int = 1800, temperature : float = 1.0):
        self._model = model
        self._llm_client = InferenceClient(model = model.value, timeout = timeout, token = token)
        
    def get_model_response(self, input_text):
        response = self._llm_client.post(
            json={
                "inputs": input_text,
                "parameters": {"max_new_tokens": 600},
                "task": "text-generation",
            },
        )
        return json.loads(response.decode())[0]["generated_text"]
    