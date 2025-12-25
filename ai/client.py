import litellm
from typing import List, Dict, Any, Optional, Union, Generator
from config import Config

class AIClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, system_instruction: Optional[str] = None, **kwargs):
        """
        Initializes the AI Client using LiteLLM.

        :param model: The default model to use (e.g., 'gemini/gemini-1.5-flash', 'gemini/gemini-pro').
        :param api_key: Optional API key. If not provided, it looks into config.yaml or environment variables.
        :param system_instruction: Optional default system instruction (system prompt) to be prepended to all requests.
        :param kwargs: Additional arguments to pass to LiteLLM's completion (e.g., temperature).
        """
        # Load from config
        config = Config()
        ai_config = config.get_ai_config()

        self.default_model = model or ai_config.get("model") or "gemini/gemini-1.5-flash"
        self.api_key = api_key or ai_config.get("api_key")
        self.system_instruction = system_instruction
        self.default_params = kwargs

    def generate_response(self, messages: List[Dict[str, Any]], model: Optional[str] = None, system_instruction: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None, stream: bool = False, **kwargs) -> Union[Any, Generator]:
        """
        Generates a response from the AI model. Supports streaming, system instructions, and multimodal inputs.

        :param messages: A list of message dictionaries. 
                         Text only: [{'role': 'user', 'content': 'Hello'}]
                         Multimodal: [{'role': 'user', 'content': [{'type': 'text', 'text': 'Describe this'}, {'type': 'image_url', 'image_url': {'url': '...'}}]}]
        :param model: Optional model override.
        :param system_instruction: Optional system instruction override. If None, uses the default from __init__.
        :param tools: Optional list of tools to enable (e.g., [{'google_search': {}}]).
        :param stream: Whether to stream the response. Defaults to False.
        :param kwargs: Optional overrides for generation parameters.
        :return: The response object from LiteLLM (or a generator if stream=True).
        """
        target_model = model or self.default_model
        # Merge default params with request-specific kwargs. 
        # kwargs overrides default_params.
        params = {**self.default_params, **kwargs}
        
        if self.api_key:
            params['api_key'] = self.api_key

        if tools:
            params['tools'] = tools

        # Handle system instruction
        active_system_instruction = system_instruction if system_instruction is not None else self.system_instruction
        final_messages = messages.copy()
        
        if active_system_instruction:
            # Check if there is already a system message at the start
            if final_messages and final_messages[0].get('role') == 'system':
                # Option A: Replace it (or you could append/prepend to it, but replacement/precedence is cleaner)
                # Option B: Just prepend ours? Standard practice is usually one system message at the start.
                # Let's prepend if the first one isn't system, or insert at 0. 
                # Actually, if the user explicitly provided a system message in `messages`, that should probably take precedence 
                # OR we treat `system_instruction` as a configuration that sits "above".
                # For simplicity and flexibility: we insert at 0. If the user sent one, they now have two (which some models handle, some don't).
                # To be safer: let's insert it at the very beginning.
                final_messages.insert(0, {"role": "system", "content": active_system_instruction})
            else:
                final_messages.insert(0, {"role": "system", "content": active_system_instruction})

        try:
            response = litellm.completion(
                model=target_model,
                messages=final_messages,
                stream=stream,
                **params
            )
            return response
        except Exception as e:
            # Propagate the exception for the caller to handle
            raise e

    def get_response_content(self, response: Any) -> Optional[str]:
        """
        Helper to extract the text content from the response object.
        Note: This is for non-streaming responses. For streaming, iterate over the response generator.
        """
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content
        return None

    @staticmethod
    def format_multimodal_message(text: str, image_urls: List[str], role: str = "user") -> Dict[str, Any]:
        """
        Helper to construct a multimodal message for Gemini/LiteLLM.
        
        :param text: The text prompt.
        :param image_urls: List of image URLs or base64 data URIs.
        :param role: The role of the message sender (default: "user").
        :return: A dictionary formatted for multimodal input.
        """
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })
        
        return {"role": role, "content": content}
