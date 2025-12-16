from abc import ABC, abstractmethod
from google import genai
from config import Config
from google.genai import types

class AIBackend(ABC):
    @abstractmethod
    def send_message(self, message, web_search: bool = False) -> str | None:
        pass

class GeminiClient(AIBackend):# Class variable to hold the client instance
    def __init__(self, ai_config):
        self.api_key = ai_config.get("api_key")
        self.model_name = ai_config.get("model", "gemini-3-pro-preview") # Default to gemini-pro if not set
        
        if not self.api_key:
            raise ValueError("API Key not found in configuration.")
            
        self.client = genai.Client(api_key=self.api_key)

    def send_message(self, message, web_search: bool = False):
        try:
            config = types.GenerateContentConfig()
            if web_search:
                grounding_tool = types.Tool(
                    google_search=types.GoogleSearch()
                )
                config = types.GenerateContentConfig(tools=[grounding_tool])
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=message,
                config=config
            )
            return response.text
        except Exception as e:
            return f"Error communicating with AI: {str(e)}"

class OpenAIClient(AIBackend):
    def __init__(self, ai_config):
        self.api_key = ai_config.get("api_key")
        self.model_name = ai_config.get("model", "gpt-3.5-turbo")
        # 这里可以初始化 OpenAI 客户端
        # import openai
        # self.client = openai.OpenAI(api_key=self.api_key)

    def send_message(self, message, web_search: bool = False):
        # 实现 OpenAI 的发送逻辑
        return "OpenAI implementation is not yet configured."

class AIProvider:
    def __init__(self):
        self.config = Config()
        self.ai_config = self.config.get_ai_config()
        self.provider_name = self.ai_config.get("provider", "gemini") # 默认为 gemini
        
        if self.provider_name == "gemini":
            self.client = GeminiClient(self.ai_config)
        elif self.provider_name == "openai":
            self.client = OpenAIClient(self.ai_config)
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider_name}")

    def send_message(self, message, web_search: bool = False):
        return self.client.send_message(message, web_search=web_search)
