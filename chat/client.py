import google.generativeai as genai
from config import Config

class GeminiClient:
    def __init__(self):
        self.config = Config()
        self.ai_config = self.config.get_ai_config()
        self.api_key = self.ai_config.get("api_key")
        self.model_name = self.ai_config.get("model", "gemini-pro") # Default to gemini-pro if not set
        
        if not self.api_key:
            raise ValueError("API Key not found in configuration.")
            
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def send_message(self, message):
        try:
            response = self.model.generate_content(message)
            return response.text
        except Exception as e:
            return f"Error communicating with AI: {str(e)}"
