"""Constants for model-specific features and limitations."""

# Models that support caching prompts
CACHE_PROMPT_SUPPORTED_MODELS = [
    'claude-3-5-sonnet-20241022',
    'claude-3-5-sonnet-20240620',
    'claude-3-5-haiku-20241022',
    'claude-3-haiku-20240307',
    'claude-3-opus-20240229',
]

# Models that support function calling
FUNCTION_CALLING_SUPPORTED_MODELS = [
    'claude-3-5-sonnet',
    'claude-3-5-sonnet-20240620',
    'claude-3-5-sonnet-20241022',
    'claude-3.5-haiku',
    'claude-3-5-haiku-20241022',
    'gpt-4o-mini',
    'gpt-4o',
    'o1-2024-12-17',
    'o3-mini-2025-01-31',
    'o3-mini',
]

# Models that support reasoning effort parameter
REASONING_EFFORT_SUPPORTED_MODELS = [
    'o1-2024-12-17',
    'o1',
    'o3-mini-2025-01-31',
    'o3-mini',
]

# Models that don't support stop words
MODELS_WITHOUT_STOP_WORDS = [
    'o1-mini',
    'o1-preview',
]

def supports_cache_prompt(model_name: str) -> bool:
    """Check if a model supports prompt caching."""
    model_base = model_name.split('/')[-1]
    return (
        model_name in CACHE_PROMPT_SUPPORTED_MODELS
        or model_base in CACHE_PROMPT_SUPPORTED_MODELS
    )

def supports_function_calling(model_name: str) -> bool:
    """Check if a model supports function calling."""
    model_base = model_name.split('/')[-1]
    return (
        model_name in FUNCTION_CALLING_SUPPORTED_MODELS
        or model_base in FUNCTION_CALLING_SUPPORTED_MODELS
        or any(m in model_name for m in FUNCTION_CALLING_SUPPORTED_MODELS)
    )

def supports_reasoning_effort(model_name: str) -> bool:
    """Check if a model supports the reasoning_effort parameter."""
    model_base = model_name.split('/')[-1].lower()
    return (
        model_name.lower() in REASONING_EFFORT_SUPPORTED_MODELS
        or model_base in REASONING_EFFORT_SUPPORTED_MODELS
    )

def supports_stop_words(model_name: str) -> bool:
    """Check if a model supports stop words."""
    model_base = model_name.split('/')[-1]
    return model_name not in MODELS_WITHOUT_STOP_WORDS and model_base not in MODELS_WITHOUT_STOP_WORDS