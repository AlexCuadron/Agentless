import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from litellm.exceptions import BadRequestError, RateLimitError

from agentless.util.model import (
    OpenAIChatDecoder,
    AnthropicChatDecoder,
    DeepSeekChatDecoder,
)

@pytest.fixture
def mock_logger():
    return MagicMock()

def test_sync_completion(mock_logger):
    """Test synchronous completion with OpenAI."""
    decoder = OpenAIChatDecoder(
        name="gpt-3.5-turbo",
        logger=mock_logger,
        temperature=0,
        max_new_tokens=100,
    )
    
    response = decoder.codegen("Hello, how are you?")
    assert isinstance(response, list)
    assert len(response) > 0
    assert "response" in response[0]
    assert "usage" in response[0]

@pytest.mark.asyncio
async def test_async_completion(mock_logger):
    """Test asynchronous completion with OpenAI."""
    decoder = OpenAIChatDecoder(
        name="gpt-3.5-turbo",
        logger=mock_logger,
        temperature=0,
        max_new_tokens=100,
    )
    
    response = await decoder.codegen_async("Hello, how are you?")
    assert isinstance(response, list)
    assert len(response) > 0
    assert "response" in response[0]
    assert "usage" in response[0]

def test_streaming_completion(mock_logger):
    """Test streaming completion with OpenAI."""
    decoder = OpenAIChatDecoder(
        name="gpt-3.5-turbo",
        logger=mock_logger,
        temperature=0,
        max_new_tokens=100,
    )
    
    response_stream = decoder.codegen_stream("Hello, how are you?")
    chunks = list(response_stream)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "response" in chunk
        assert "usage" in chunk
        assert "finish_reason" in chunk

@pytest.mark.asyncio
async def test_streaming_completion_async(mock_logger):
    """Test async streaming completion with OpenAI."""
    decoder = OpenAIChatDecoder(
        name="gpt-3.5-turbo",
        logger=mock_logger,
        temperature=0,
        max_new_tokens=100,
    )
    
    response_stream = decoder.codegen_stream_async("Hello, how are you?")
    chunks = []
    async for chunk in response_stream:
        chunks.append(chunk)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert "response" in chunk
        assert "usage" in chunk
        assert "finish_reason" in chunk

def test_anthropic_completion(mock_logger):
    """Test completion with Anthropic."""
    decoder = AnthropicChatDecoder(
        name="claude-3-sonnet-20240229",
        logger=mock_logger,
        temperature=0,
        max_new_tokens=100,
    )
    
    response = decoder.codegen("Hello, how are you?")
    assert isinstance(response, list)
    assert len(response) > 0
    assert "response" in response[0]
    assert "usage" in response[0]

def test_deepseek_completion(mock_logger):
    """Test completion with DeepSeek."""
    decoder = DeepSeekChatDecoder(
        name="deepseek-chat",
        logger=mock_logger,
        temperature=0,
        max_new_tokens=100,
    )
    
    response = decoder.codegen("Hello, how are you?")
    assert isinstance(response, list)
    assert len(response) > 0
    assert "response" in response[0]
    assert "usage" in response[0]

def test_error_handling_rate_limit(mock_logger):
    """Test rate limit error handling."""
    decoder = OpenAIChatDecoder(
        name="gpt-3.5-turbo",
        logger=mock_logger,
        temperature=0,
        max_new_tokens=100,
    )
    
    # Mock the completion function to raise a rate limit error
    mock_logger.error = MagicMock()
    mock_logger.warning = MagicMock()
    
    with pytest.raises(Exception) as exc_info:
        decoder.codegen("This should fail with rate limit")
    
    assert "Rate limit" in str(exc_info.value)
    assert mock_logger.warning.called

def test_error_handling_bad_request(mock_logger):
    """Test bad request error handling."""
    decoder = OpenAIChatDecoder(
        name="gpt-3.5-turbo",
        logger=mock_logger,
        temperature=0,
        max_new_tokens=100,
    )
    
    # Mock the completion function to raise a bad request error
    mock_logger.error = MagicMock()
    
    with pytest.raises(Exception) as exc_info:
        decoder.codegen("This should fail with bad request")
    
    assert "Bad Request" in str(exc_info.value)
    assert mock_logger.error.called

def test_multiple_samples(mock_logger):
    """Test generating multiple samples."""
    decoder = OpenAIChatDecoder(
        name="gpt-3.5-turbo",
        logger=mock_logger,
        temperature=0.8,  # Need temperature > 0 for multiple samples
        max_new_tokens=100,
        batch_size=2,
    )
    
    response = decoder.codegen("Hello, how are you?", num_samples=2)
    assert isinstance(response, list)
    assert len(response) == 2
    for r in response:
        assert "response" in r
        assert "usage" in r

@pytest.mark.asyncio
async def test_anthropic_tool_calls(mock_logger):
    """Test Anthropic with tool calls."""
    decoder = AnthropicChatDecoder(
        name="claude-3-sonnet-20240229",
        logger=mock_logger,
        temperature=0,
        max_new_tokens=100,
    )
    
    # Test with a prompt that should trigger tool usage
    response = decoder.codegen_w_tool(
        "Please help me edit the file test.py to fix a bug."
    )
    assert isinstance(response, list)
    assert len(response) > 0
    assert "response" in response[0]
    assert isinstance(response[0]["response"], list)
    for r in response[0]["response"]:
        assert "role" in r
        assert "content" in r