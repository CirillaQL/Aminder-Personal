from abc import ABC, abstractmethod
from typing import Optional
from chat.client import GeminiClient

class AIProvider(ABC):
    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        pass

class GeminiAIProvider(AIProvider):
    def __init__(self):
        self.client = GeminiClient()

    async def generate_response(self, prompt: str) -> str:
        return self.client.send_message(prompt)

class AIProviderFactory:
    _instance: Optional[AIProvider] = None

    @staticmethod
    def get_provider() -> AIProvider:
        if AIProviderFactory._instance is None:
            # For now, we only have Gemini. Can add logic here to select provider based on config.
            AIProviderFactory._instance = GeminiAIProvider()
        return AIProviderFactory._instance
