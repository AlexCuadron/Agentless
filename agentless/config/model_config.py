from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class ModelConfig:
    """Configuration for a model in Agentless."""
    name: str
    provider: str  # openai, anthropic, deepseek, etc.
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_new_tokens: int = 1024
    batch_size: int = 1
    num_retries: int = 3
    retry_min_wait: int = 4
    retry_max_wait: int = 10
    retry_multiplier: float = 2
    tools: Optional[List[Dict[str, Any]]] = None
    prompt_cache: bool = False
    stream: bool = False

@dataclass
class AgentlessConfig:
    """Main configuration for Agentless."""
    models: Dict[str, ModelConfig]
    default_model: str
    logging_config: Optional[Dict[str, Any]] = None