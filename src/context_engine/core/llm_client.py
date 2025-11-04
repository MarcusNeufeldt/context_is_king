"""
LLM Client for OpenRouter integration.

Provides a clean interface to interact with LLMs via OpenRouter,
with support for streaming, token counting, and error handling.
"""

import os
from typing import List, Dict, Any, Optional, Iterator
from pydantic import BaseModel, Field
from openai import OpenAI
import tiktoken

from ..utils.schemas import Message, MessageRole


class LLMConfig(BaseModel):
    """Configuration for LLM client."""
    model: str = "google/gemini-2.5-flash"
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0

    # OpenRouter specific
    site_url: Optional[str] = None
    site_name: Optional[str] = None

    # Token management
    context_window: int = 128000  # Gemini 2.5 Flash context window
    max_context_usage: float = 0.5  # Keep context <50% full (paper recommendation)


class LLMClient:
    """
    Client for interacting with LLMs via OpenRouter.

    Features:
    - Streaming and non-streaming responses
    - Token counting and context window management
    - Error handling and retry logic
    - KV cache optimization (stable prefixes)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM client with configuration."""
        self.config = config or LLMConfig()

        # Load from environment if not provided
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.config.site_url:
            self.config.site_url = os.getenv("OPENROUTER_SITE_URL")

        if not self.config.site_name:
            self.config.site_name = os.getenv("OPENROUTER_SITE_NAME")

        # Initialize OpenAI client (OpenRouter is compatible)
        extra_headers = {}
        if self.config.site_url:
            extra_headers["HTTP-Referer"] = self.config.site_url
        if self.config.site_name:
            extra_headers["X-Title"] = self.config.site_name

        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            default_headers=extra_headers
        )

        # Token counter (fallback to cl100k_base for estimation)
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages: List[Message]) -> int:
        """
        Count tokens in a list of messages.

        Uses tiktoken for estimation. Note: actual token count may vary
        slightly depending on the model's tokenizer.
        """
        total = 0
        for msg in messages:
            # Count role and content
            total += len(self.tokenizer.encode(msg.role.value))
            total += len(self.tokenizer.encode(msg.content))
            # Add overhead for message formatting (~4 tokens per message)
            total += 4
        return total

    def check_context_limit(self, messages: List[Message]) -> tuple[bool, int]:
        """
        Check if messages fit within context window limits.

        Returns:
            (fits_in_window, token_count)
        """
        token_count = self.count_tokens(messages)
        max_allowed = int(self.config.context_window * self.config.max_context_usage)
        return token_count <= max_allowed, token_count

    def complete(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> tuple[str, Dict[str, Any]]:
        """
        Get a completion from the LLM (non-streaming).

        Returns:
            (response_content, usage_stats)
        """
        # Convert messages to OpenAI format
        openai_messages = [msg.to_openai_format() for msg in messages]

        # Make API call
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=openai_messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            top_p=self.config.top_p,
            **kwargs
        )

        # Extract response
        content = response.choices[0].message.content

        # Usage statistics
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
            "model": response.model,
        }

        return content, usage

    def complete_stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Get a streaming completion from the LLM.

        Yields:
            Content chunks as they arrive
        """
        # Convert messages to OpenAI format
        openai_messages = [msg.to_openai_format() for msg in messages]

        # Make streaming API call
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=openai_messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            top_p=self.config.top_p,
            stream=True,
            **kwargs
        )

        # Stream chunks
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def create_system_message(self, content: str) -> Message:
        """Create a system message with stable formatting for KV cache."""
        return Message(role=MessageRole.SYSTEM, content=content)

    def create_user_message(self, content: str) -> Message:
        """Create a user message."""
        return Message(role=MessageRole.USER, content=content)

    def create_assistant_message(self, content: str) -> Message:
        """Create an assistant message."""
        return Message(role=MessageRole.ASSISTANT, content=content)

    def optimize_for_kv_cache(self, messages: List[Message]) -> List[Message]:
        """
        Optimize message list for KV cache efficiency.

        Key strategy: Keep system prompts stable and at the beginning.
        Avoid dynamic prefixes that invalidate the cache.
        """
        # Separate system messages from conversation
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        # System messages go first (stable prefix for cache)
        return system_messages + other_messages
