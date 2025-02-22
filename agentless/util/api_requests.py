import asyncio
import time
from typing import Dict, List, Union, Optional

import tiktoken
from litellm import completion, acompletion
from litellm.exceptions import (
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    InvalidRequestError,
    AuthenticationError,
)


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100, stream=False):
    """
    Make a request to any LLM provider using LiteLLM's completion function.
    
    Args:
        config (dict): The configuration for the request
        logger: Logger object
        base_url (str, optional): Base URL for the API. Defaults to None.
        max_retries (int, optional): Maximum number of retries. Defaults to 40.
        timeout (int, optional): Timeout in seconds. Defaults to 100.
        stream (bool, optional): Whether to stream the response. Defaults to False.
    """
    ret = None
    retries = 0
    start_time = time.time()

    model = config.pop("model")
    # Map provider prefixes based on base_url or model name
    if base_url == "https://api.deepseek.com":
        model = f"deepseek/{model}"
    elif base_url and "azure" in base_url:
        model = f"azure/{model}"
    elif base_url and "vertex.ai" in base_url:
        model = f"vertex_ai/{model}"
    else:
        model = f"openai/{model}"

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info(f"Creating API request for model {model}")
            
            ret = completion(
                model=model,
                api_base=base_url,
                stream=stream,
                **config
            )

        except BadRequestError as e:
            logger.error("Bad request error", exc_info=True)
            raise Exception(f"Bad Request: {str(e)}")
        
        except InvalidRequestError as e:
            logger.error("Invalid request error", exc_info=True)
            raise Exception(f"Invalid Request: {str(e)}")
        
        except RateLimitError as e:
            wait_time = min(5 * (retries + 1), 60)  # Exponential backoff up to 60s
            logger.warning(f"Rate limit exceeded. Waiting {wait_time}s...", exc_info=True)
            time.sleep(wait_time)
        
        except ServiceUnavailableError as e:
            wait_time = min(5 * (retries + 1), 30)  # Exponential backoff up to 30s
            logger.warning(f"Service unavailable. Waiting {wait_time}s...", exc_info=True)
            time.sleep(wait_time)
        
        except AuthenticationError as e:
            logger.error("Authentication error", exc_info=True)
            raise Exception(f"Authentication Error: {str(e)}")
        
        except Exception as e:
            # Check if timeout exceeded
            if time.time() - start_time >= timeout:
                logger.error("Request timeout exceeded", exc_info=True)
                raise Exception(f"Timeout exceeded after {timeout}s")
            
            wait_time = min(2 * (retries + 1), 20)  # Exponential backoff up to 20s
            logger.warning(f"Unknown error. Waiting {wait_time}s...", exc_info=True)
            time.sleep(wait_time)

        retries += 1

    if ret is None:
        raise Exception(f"Failed to get response after {max_retries} retries")

    logger.info(f"API response received after {retries} tries")
    return ret


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
    tools: list = None,
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

    if tools:
        config["tools"] = tools

    return config


async def request_chatgpt_engine_async(
    config, logger, base_url=None, max_retries=40, timeout=100, stream=False
):
    """
    Async version of request_chatgpt_engine using LiteLLM's acompletion function.
    """
    ret = None
    retries = 0
    start_time = time.time()

    model = config.pop("model")
    # Map provider prefixes based on base_url or model name
    if base_url == "https://api.deepseek.com":
        model = f"deepseek/{model}"
    elif base_url and "azure" in base_url:
        model = f"azure/{model}"
    elif base_url and "vertex.ai" in base_url:
        model = f"vertex_ai/{model}"
    else:
        model = f"openai/{model}"

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info(f"Creating async API request for model {model}")
            
            ret = await acompletion(
                model=model,
                api_base=base_url,
                stream=stream,
                **config
            )

        except BadRequestError as e:
            logger.error("Bad request error", exc_info=True)
            raise Exception(f"Bad Request: {str(e)}")
        
        except InvalidRequestError as e:
            logger.error("Invalid request error", exc_info=True)
            raise Exception(f"Invalid Request: {str(e)}")
        
        except RateLimitError as e:
            wait_time = min(5 * (retries + 1), 60)  # Exponential backoff up to 60s
            logger.warning(f"Rate limit exceeded. Waiting {wait_time}s...", exc_info=True)
            await asyncio.sleep(wait_time)
        
        except ServiceUnavailableError as e:
            wait_time = min(5 * (retries + 1), 30)  # Exponential backoff up to 30s
            logger.warning(f"Service unavailable. Waiting {wait_time}s...", exc_info=True)
            await asyncio.sleep(wait_time)
        
        except AuthenticationError as e:
            logger.error("Authentication error", exc_info=True)
            raise Exception(f"Authentication Error: {str(e)}")
        
        except Exception as e:
            # Check if timeout exceeded
            if time.time() - start_time >= timeout:
                logger.error("Request timeout exceeded", exc_info=True)
                raise Exception(f"Timeout exceeded after {timeout}s")
            
            wait_time = min(2 * (retries + 1), 20)  # Exponential backoff up to 20s
            logger.warning(f"Unknown error. Waiting {wait_time}s...", exc_info=True)
            await asyncio.sleep(wait_time)

        retries += 1

    if ret is None:
        raise Exception(f"Failed to get response after {max_retries} retries")

    logger.info(f"API response received after {retries} tries")
    return ret


async def request_anthropic_engine_async(
    config, logger, max_retries=40, timeout=500, prompt_cache=False, stream=False
):
    """
    Async version of request_anthropic_engine using LiteLLM's acompletion function.
    """
    ret = None
    retries = 0
    start_time = time.time()

    model = f"anthropic/{config.pop('model')}"

    while ret is None and retries < max_retries:
        try:
            if prompt_cache:
                # following best practice to cache mainly the reused content at the beginning
                # this includes any tools, system messages (which is already handled since we try to cache the first message)
                if isinstance(config["messages"][0]["content"], list):
                    config["messages"][0]["content"][0]["cache_control"] = {
                        "type": "ephemeral"
                    }
            
            ret = await acompletion(
                model=model,
                stream=stream,
                **config
            )

        except BadRequestError as e:
            logger.error("Bad request error", exc_info=True)
            raise Exception(f"Bad Request: {str(e)}")
        
        except InvalidRequestError as e:
            logger.error("Invalid request error", exc_info=True)
            raise Exception(f"Invalid Request: {str(e)}")
        
        except RateLimitError as e:
            wait_time = min(5 * (retries + 1), 60)  # Exponential backoff up to 60s
            logger.warning(f"Rate limit exceeded. Waiting {wait_time}s...", exc_info=True)
            await asyncio.sleep(wait_time)
        
        except ServiceUnavailableError as e:
            wait_time = min(5 * (retries + 1), 30)  # Exponential backoff up to 30s
            logger.warning(f"Service unavailable. Waiting {wait_time}s...", exc_info=True)
            await asyncio.sleep(wait_time)
        
        except AuthenticationError as e:
            logger.error("Authentication error", exc_info=True)
            raise Exception(f"Authentication Error: {str(e)}")
        
        except Exception as e:
            # Check if timeout exceeded
            if time.time() - start_time >= timeout:
                logger.error("Request timeout exceeded", exc_info=True)
                raise Exception(f"Timeout exceeded after {timeout}s")
            
            wait_time = min(10 * (retries + 1), 60)  # Exponential backoff up to 60s
            logger.warning(f"Unknown error. Waiting {wait_time}s...", exc_info=True)
            await asyncio.sleep(wait_time)

        retries += 1

    if ret is None:
        raise Exception(f"Failed to get response after {max_retries} retries")

    logger.info(f"API response received after {retries} tries")
    return ret


def request_anthropic_engine(
    config, logger, max_retries=40, timeout=500, prompt_cache=False, stream=False
):
    """
    Make a request to Anthropic's API using LiteLLM's completion function.
    
    Args:
        config (dict): The configuration for the request
        logger: Logger object
        max_retries (int, optional): Maximum number of retries. Defaults to 40.
        timeout (int, optional): Timeout in seconds. Defaults to 500.
        prompt_cache (bool, optional): Whether to use prompt caching. Defaults to False.
        stream (bool, optional): Whether to stream the response. Defaults to False.
    """
    ret = None
    retries = 0
    start_time = time.time()

    model = f"anthropic/{config.pop('model')}"

    while ret is None and retries < max_retries:
        try:
            if prompt_cache:
                # following best practice to cache mainly the reused content at the beginning
                # this includes any tools, system messages (which is already handled since we try to cache the first message)
                if isinstance(config["messages"][0]["content"], list):
                    config["messages"][0]["content"][0]["cache_control"] = {
                        "type": "ephemeral"
                    }
            
            ret = completion(
                model=model,
                stream=stream,
                **config
            )

        except BadRequestError as e:
            logger.error("Bad request error", exc_info=True)
            raise Exception(f"Bad Request: {str(e)}")
        
        except InvalidRequestError as e:
            logger.error("Invalid request error", exc_info=True)
            raise Exception(f"Invalid Request: {str(e)}")
        
        except RateLimitError as e:
            wait_time = min(5 * (retries + 1), 60)  # Exponential backoff up to 60s
            logger.warning(f"Rate limit exceeded. Waiting {wait_time}s...", exc_info=True)
            time.sleep(wait_time)
        
        except ServiceUnavailableError as e:
            wait_time = min(5 * (retries + 1), 30)  # Exponential backoff up to 30s
            logger.warning(f"Service unavailable. Waiting {wait_time}s...", exc_info=True)
            time.sleep(wait_time)
        
        except AuthenticationError as e:
            logger.error("Authentication error", exc_info=True)
            raise Exception(f"Authentication Error: {str(e)}")
        
        except Exception as e:
            # Check if timeout exceeded
            if time.time() - start_time >= timeout:
                logger.error("Request timeout exceeded", exc_info=True)
                raise Exception(f"Timeout exceeded after {timeout}s")
            
            wait_time = min(10 * (retries + 1), 60)  # Exponential backoff up to 60s
            logger.warning(f"Unknown error. Waiting {wait_time}s...", exc_info=True)
            time.sleep(wait_time)

        retries += 1

    if ret is None:
        raise Exception(f"Failed to get response after {max_retries} retries")

    logger.info(f"API response received after {retries} tries")
    return ret
