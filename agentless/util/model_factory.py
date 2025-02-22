from typing import Dict, Optional
import logging

from ..config.model_config import AgentlessConfig, ModelConfig
from .model import OpenAIChatDecoder, AnthropicChatDecoder, DeepSeekChatDecoder, DecoderBase

class ModelFactory:
    """Factory class for creating model instances."""
    
    def __init__(self, config: AgentlessConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        if config.logging_config:
            logging.basicConfig(**config.logging_config)
    
    def create_model(self, model_name: Optional[str] = None) -> DecoderBase:
        """Create a model instance based on configuration.
        
        Args:
            model_name: Name of the model to create. If None, uses default_model from config.
        
        Returns:
            An instance of a DecoderBase subclass.
        
        Raises:
            ValueError: If model_name is not found in config or provider is not supported.
        """
        # Use default model if none specified
        if model_name is None:
            model_name = self.config.default_model
        
        # Get model config
        if model_name not in self.config.models:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        model_config = self.config.models[model_name]
        
        # Create model based on provider
        if model_config.provider == "openai":
            return OpenAIChatDecoder(config=model_config, logger=self.logger)
        elif model_config.provider == "anthropic":
            return AnthropicChatDecoder(config=model_config, logger=self.logger)
        elif model_config.provider == "deepseek":
            return DeepSeekChatDecoder(config=model_config, logger=self.logger)
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
    
    def list_available_models(self) -> Dict[str, str]:
        """List all available models and their providers."""
        return {
            name: config.provider 
            for name, config in self.config.models.items()
        }