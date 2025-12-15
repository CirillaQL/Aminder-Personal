from abc import ABC, abstractmethod
import google.generativeai as genai
from config import Config

class AIBackend(ABC):
    @abstractmethod
    def send_message(self, message) -> str:
        pass

class GeminiClient(AIBackend):
    def __init__(self, ai_config):
        self.api_key = ai_config.get("api_key")
        self.model_name = ai_config.get("model", "gemini-pro") # Default to gemini-pro if not set
        
        if not self.api_key:
            raise ValueError("API Key not found in configuration.")
            
        genai.configure(api_key=self.api_key) # type: ignore
        self.model = genai.GenerativeModel(self.model_name) # type: ignore

    def send_message(self, message):
        try:
            response = self.model.generate_content(message)
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

    def send_message(self, message):
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

    def send_message(self, message):
        return self.client.send_message(message)
