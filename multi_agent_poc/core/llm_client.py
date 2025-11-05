"""
LLM Client Module - OpenRouter Integration

Provides a unified interface for LLM calls using OpenRouter API
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import urllib.request
import urllib.error


class OpenRouterClient:
    """
    Client for OpenRouter API supporting multiple model providers.

    Uses the API key from .env file as specified in the environment.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL", "")
        self.site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", "Multi-Agent POC")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "minimax/minimax-m2",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenRouter.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (e.g., 'minimax/minimax-m2')
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Response dict with 'choices', 'usage', etc.
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json"
        }

        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(request) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise Exception(f"OpenRouter API error: {e.code} - {error_body}")

        except Exception as e:
            raise Exception(f"Failed to call OpenRouter API: {str(e)}")

    def get_assistant_message(self, response: Dict[str, Any]) -> str:
        """Extract the assistant's message from the API response"""
        try:
            return response['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid response format: {e}")

    def stream_chat_completion(self, messages: List[Dict[str, str]], **kwargs):
        """
        Stream a chat completion (future enhancement).
        For now, returns regular completion.
        """
        # Streaming support can be added later
        return self.chat_completion(messages, **kwargs)


def load_env_from_file(env_path: Path):
    """
    Load environment variables from a .env file.

    This is a simple implementation for environments without python-dotenv.
    """
    if not env_path.exists():
        return

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


# Load .env on module import if present
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    load_env_from_file(env_file)
