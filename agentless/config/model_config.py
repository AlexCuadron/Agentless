from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from .model_constants import (
    supports_cache_prompt,
    supports_function_calling,
    supports_reasoning_effort,
    supports_stop_words,
)

@dataclass
class ModelConfig:
    """Configuration for a model in Agentless."""
    name: str
    provider: str  # openai, anthropic, deepseek, etc.
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    custom_llm_provider: Optional[str] = None
    
    # Model parameters
    temperature: float = 0.0
    max_new_tokens: int = 1024  # max_completion_tokens in LiteLLM
    max_input_tokens: Optional[int] = None
    batch_size: int = 1
    top_p: float = 1.0
    reasoning_effort: Optional[float] = None  # Only for supported models
    
    # Retry configuration
    num_retries: int = 3
    retry_min_wait: int = 4
    retry_max_wait: int = 10
    retry_multiplier: float = 2
    timeout: int = 100
    
    # Features
    tools: Optional[List[Dict[str, Any]]] = None
    prompt_cache: bool = False
    stream: bool = False
    drop_params: bool = True  # Whether to drop unsupported params
    modify_params: bool = True  # Allow LiteLLM to modify params
    stop: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate and adjust configuration based on model capabilities."""
        # Check if model supports prompt caching
        if self.prompt_cache and not supports_cache_prompt(self.name):
            self.prompt_cache = False
        
        # Check if model supports function calling
        if self.tools and not supports_function_calling(self.name):
            self.tools = None
        
        # Check if model supports reasoning effort
        if self.reasoning_effort is not None:
            if supports_reasoning_effort(self.name):
                # Remove temperature for reasoning models
                self.temperature = 0.0
            else:
                self.reasoning_effort = None
        
        # Check if model supports stop words
        if self.stop and not supports_stop_words(self.name):
            self.stop = None
        
        # Special handling for Hugging Face models
        if self.provider == "huggingface":
            # HF doesn't support the OpenAI default value for top_p (1)
            self.top_p = 0.9 if self.top_p == 1 else self.top_p
        
        # Set default max tokens if not set
        if self.max_input_tokens is None:
            self.max_input_tokens = 4096  # Safe default
        
    def to_completion_kwargs(self) -> Dict[str, Any]:
        """Convert config to kwargs for LiteLLM completion."""
        kwargs = {
            "model": self.name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_version": self.api_version,
            "custom_llm_provider": self.custom_llm_provider,
            "max_completion_tokens": self.max_new_tokens,
            "timeout": self.timeout,
            "top_p": self.top_p,
            "drop_params": self.drop_params,
            "stream": self.stream,
        }
        
        # Add optional parameters
        if not supports_reasoning_effort(self.name):
            kwargs["temperature"] = self.temperature
        else:
            kwargs["reasoning_effort"] = self.reasoning_effort
        
        if self.tools:
            kwargs["tools"] = self.tools
        
        if self.stop:
            kwargs["stop"] = self.stop
        
        # Remove None values
        return {k: v for k, v in kwargs.items() if v is not None}

@dataclass
class AgentlessConfig:
    """Main configuration for Agentless."""
    models: Dict[str, ModelConfig]
    default_model: str
    logging_config: Optional[Dict[str, Any]] = None