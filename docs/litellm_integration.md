# LiteLLM Integration in Agentless

This document describes the LiteLLM integration in Agentless, including configuration, features, and model-specific considerations.

## Overview

Agentless uses LiteLLM to provide a unified interface to multiple LLM providers while maintaining consistent behavior and error handling. The integration includes:

- Async and streaming support for all operations
- TOML-based configuration system
- Model-specific feature detection
- Comprehensive error handling
- Retry mechanisms with exponential backoff

## Configuration System

### Basic Structure

The configuration uses TOML format and is divided into sections:

```toml
# Default model to use
default_model = "gpt-4"

# Model configurations
[models.gpt-4]
name = "gpt-4"
provider = "openai"
# ... model-specific settings

# Logging configuration
[logging]
level = "INFO"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Model Configuration Options

Each model can be configured with the following parameters:

#### Basic Settings
- `name`: Model name (required)
- `provider`: Provider name (required) - openai, anthropic, deepseek, etc.
- `api_key`: API key (can use environment variables with "env:" prefix)
- `base_url`: Custom API endpoint URL
- `api_version`: API version for providers that support it
- `custom_llm_provider`: Custom provider name

#### Model Parameters
- `temperature`: Temperature for sampling (0.0 to 1.0)
- `max_new_tokens`: Maximum tokens in response
- `max_input_tokens`: Maximum tokens in input
- `batch_size`: Number of completions to generate
- `top_p`: Top-p sampling parameter
- `reasoning_effort`: For models that support it (e.g., O3-mini)

#### Retry Configuration
- `num_retries`: Number of retries on failure
- `retry_min_wait`: Minimum wait time between retries
- `retry_max_wait`: Maximum wait time between retries
- `retry_multiplier`: Multiplier for exponential backoff
- `timeout`: Request timeout in seconds

#### Features
- `prompt_cache`: Enable prompt caching (supported models only)
- `stream`: Enable streaming responses
- `drop_params`: Drop unsupported parameters
- `modify_params`: Allow LiteLLM to modify parameters
- `stop`: Optional stop words
- `tools`: Function calling configuration (supported models only)

### Model-Specific Features

Different models support different features. The configuration system automatically handles these differences:

#### Prompt Caching
Supported by:
- Claude-3 Sonnet (20241022, 20240620)
- Claude-3 Haiku (20241022, 20240307)
- Claude-3 Opus (20240229)

#### Function Calling
Supported by:
- Claude-3 models
- GPT-4 and variants
- O1 and O3 models

#### Reasoning Effort
Supported by:
- O1 models
- O3-mini models

Note: Models that support reasoning_effort don't use temperature.

#### Stop Words
Not supported by:
- O1-mini
- O1-preview

### Example Configurations

#### GPT-4 with Standard Settings
```toml
[models.gpt-4]
name = "gpt-4"
provider = "openai"
api_key = "env:OPENAI_API_KEY"
temperature = 0.0
max_new_tokens = 1024
max_input_tokens = 8192
```

#### Claude-3 with Function Calling
```toml
[models.claude-3]
name = "claude-3-sonnet-20240229"
provider = "anthropic"
api_key = "env:ANTHROPIC_API_KEY"
temperature = 0.0
max_new_tokens = 4096
max_input_tokens = 200000
prompt_cache = true
tools = [
    { type = "function", function = { name = "str_replace_editor", ... } }
]
```

#### O3-mini with Reasoning Effort
```toml
[models.o3-mini]
name = "o3-mini"
provider = "openai"
api_key = "env:OPENAI_API_KEY"
reasoning_effort = 0.5
max_new_tokens = 4096
```

## Usage Examples

### Basic Usage
```python
from agentless.config import load_config
from agentless.util.model_factory import ModelFactory

# Load configuration
config = load_config("config.toml")

# Create model factory
factory = ModelFactory(config)

# Get model instance
model = factory.create_model("gpt-4")

# Use model
response = model.codegen("Hello, world!")
```

### Async Usage
```python
# Async completion
response = await model.codegen_async("Hello, world!")

# Async streaming
async for chunk in model.codegen_stream_async("Hello, world!"):
    print(chunk["response"])
```

### Streaming Usage
```python
# Synchronous streaming
for chunk in model.codegen_stream("Hello, world!"):
    print(chunk["response"])
```

### Function Calling
```python
# Configure tools in config.toml
tools = [
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "Custom editing tool for editing files",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"description": "File path", "type": "string"},
                    "old_str": {"description": "String to replace", "type": "string"},
                    "new_str": {"description": "New string", "type": "string"}
                },
                "required": ["path", "old_str"]
            }
        }
    }
]

# Use model with tools
response = model.codegen("Edit the file to fix the bug")
```

## Error Handling

The integration includes comprehensive error handling:

- `BadRequestError`: Invalid request parameters
- `RateLimitError`: Rate limit exceeded
- `ServiceUnavailableError`: Service temporarily unavailable
- `InvalidRequestError`: Invalid request format
- `AuthenticationError`: Invalid API key or credentials

Errors are handled with exponential backoff retry:
```python
# Example retry configuration
num_retries = 3
retry_min_wait = 4  # seconds
retry_max_wait = 10  # seconds
retry_multiplier = 2  # exponential backoff multiplier
```

## Testing

The integration includes comprehensive tests:

- Sync/async completion tests
- Streaming response tests
- Error handling tests
- Model-specific feature tests
- Configuration validation tests

Run tests with:
```bash
pytest agentless/test/test_litellm_integration.py
```