import os
from typing import Dict, Any
import tomli
from dotenv import load_dotenv

from .model_config import ModelConfig, AgentlessConfig

def load_config(config_path: str = "config.toml") -> AgentlessConfig:
    """Load configuration from config.toml and environment variables."""
    # Load environment variables
    load_dotenv()
    
    # Load config.toml
    with open(config_path, "rb") as f:
        config = tomli.load(f)
    
    # Create model configs
    models: Dict[str, ModelConfig] = {}
    for model_name, model_config in config.get("models", {}).items():
        # Get API key from environment if specified with env:
        api_key = model_config.get("api_key", "")
        if api_key.startswith("env:"):
            env_var = api_key.split(":", 1)[1]
            api_key = os.getenv(env_var, "")
        
        # Create ModelConfig instance
        models[model_name] = ModelConfig(
            name=model_config["name"],
            provider=model_config["provider"],
            api_key=api_key,
            base_url=model_config.get("base_url"),
            temperature=model_config.get("temperature", 0.0),
            max_new_tokens=model_config.get("max_new_tokens", 1024),
            batch_size=model_config.get("batch_size", 1),
            num_retries=model_config.get("num_retries", 3),
            retry_min_wait=model_config.get("retry_min_wait", 4),
            retry_max_wait=model_config.get("retry_max_wait", 10),
            retry_multiplier=model_config.get("retry_multiplier", 2),
            tools=model_config.get("tools"),
            prompt_cache=model_config.get("prompt_cache", False),
            stream=model_config.get("stream", False)
        )
    
    # Create AgentlessConfig instance
    return AgentlessConfig(
        models=models,
        default_model=config.get("default_model", next(iter(models.keys()))),
        logging_config=config.get("logging", {})
    )