import json
from abc import ABC, abstractmethod
from typing import List, Union, AsyncIterator, Iterator

from agentless.util.api_requests import (
    create_anthropic_config,
    create_chatgpt_config,
    request_anthropic_engine,
    request_chatgpt_engine,
    request_anthropic_engine_async,
    request_chatgpt_engine_async,
)


from ..config.model_config import ModelConfig

class DecoderBase(ABC):
    def __init__(
        self,
        config: ModelConfig,
        logger,
    ) -> None:
        logger.info("Initializing a decoder model: {} ...".format(config.name))
        self.name = config.name
        self.logger = logger
        self.batch_size = config.batch_size
        self.temperature = config.temperature
        self.max_new_tokens = config.max_new_tokens
        self.config = config

    @abstractmethod
    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        pass

    @abstractmethod
    async def codegen_async(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        pass

    @abstractmethod
    def codegen_stream(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> Iterator[dict]:
        pass

    @abstractmethod
    async def codegen_stream_async(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> AsyncIterator[dict]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def _create_trajectory(self, responses: List[str], completion_tokens: int, prompt_tokens: int) -> List[dict]:
        """Helper method to create trajectory objects from responses and token counts."""
        # The nice thing is, when we generate multiple samples from the same input (message),
        # the input tokens are only charged once according to openai API.
        # Therefore, we assume the request cost is only counted for the first sample.
        # More specifically, the `prompt_tokens` is for one input message,
        # and the `completion_tokens` is the sum of all returned completions.
        # Therefore, for the second and later samples, the cost is zero.
        trajs = [
            {
                "response": responses[0],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
        ]
        for response in responses[1:]:
            trajs.append(
                {
                    "response": response,
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                    },
                }
            )
        return trajs

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
        )
        ret = request_chatgpt_engine(config, self.logger)
        if ret:
            responses = [choice.message.content for choice in ret.choices]
            completion_tokens = ret.usage.completion_tokens
            prompt_tokens = ret.usage.prompt_tokens
        else:
            responses = [""]
            completion_tokens = 0
            prompt_tokens = 0

        return self._create_trajectory(responses, completion_tokens, prompt_tokens)

    async def codegen_async(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
        )
        ret = await request_chatgpt_engine_async(config, self.logger)
        if ret:
            responses = [choice.message.content for choice in ret.choices]
            completion_tokens = ret.usage.completion_tokens
            prompt_tokens = ret.usage.prompt_tokens
        else:
            responses = [""]
            completion_tokens = 0
            prompt_tokens = 0

        return self._create_trajectory(responses, completion_tokens, prompt_tokens)

    def codegen_stream(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> Iterator[dict]:
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
        )
        response_stream = request_chatgpt_engine(config, self.logger, stream=True)
        
        for chunk in response_stream:
            if chunk and chunk.choices:
                yield {
                    "response": chunk.choices[0].delta.content or "",
                    "usage": {
                        "completion_tokens": 0,  # Token counts only available in final chunk
                        "prompt_tokens": 0,
                    },
                    "finish_reason": chunk.choices[0].finish_reason,
                }

    async def codegen_stream_async(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> AsyncIterator[dict]:
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
        )
        response_stream = await request_chatgpt_engine_async(config, self.logger, stream=True)
        
        async for chunk in response_stream:
            if chunk and chunk.choices:
                yield {
                    "response": chunk.choices[0].delta.content or "",
                    "usage": {
                        "completion_tokens": 0,  # Token counts only available in final chunk
                        "prompt_tokens": 0,
                    },
                    "finish_reason": chunk.choices[0].finish_reason,
                }

    def is_direct_completion(self) -> bool:
        return False


class AnthropicChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    _STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for editing files
* State is persistent across command calls and discussions with the user

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

    _USER_REPLY_EDIT_MESSAGE = """File is successfully edited"""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": _STR_REPLACE_EDITOR_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "description": "Full path to file, e.g. `folder/file.py`.",
                            "type": "string",
                        },
                        "old_str": {
                            "description": "Required parameter containing the string in `path` to replace.",
                            "type": "string",
                        },
                        "new_str": {
                            "description": "Optional parameter containing the new string (if not given, no string will be added).",
                            "type": "string",
                        },
                    },
                    "required": ["path", "old_str"],
                },
            },
        }
    ]

    MAX_CODEGEN_ITERATIONS = 10

    def _create_trajectory(self, response, completion_tokens: int, prompt_tokens: int) -> dict:
        """Helper method to create a trajectory object from a response and token counts."""
        return {
            "response": response,
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "cache_creation_token": 0,
                "cache_read_input_tokens": 0,
            },
        }

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_anthropic_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_anthropic_engine(
                config, self.logger, prompt_cache=prompt_cache
            )

            if ret:
                trajs.append(self._create_trajectory(
                    ret.choices[0].message.content,
                    ret.usage.completion_tokens,
                    ret.usage.prompt_tokens
                ))
            else:
                trajs.append(self._create_trajectory("", 0, 0))

        return trajs

    async def codegen_async(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_anthropic_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = await request_anthropic_engine_async(
                config, self.logger, prompt_cache=prompt_cache
            )

            if ret:
                trajs.append(self._create_trajectory(
                    ret.choices[0].message.content,
                    ret.usage.completion_tokens,
                    ret.usage.prompt_tokens
                ))
            else:
                trajs.append(self._create_trajectory("", 0, 0))

        return trajs

    def codegen_stream(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> Iterator[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        config = create_anthropic_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=1,
            model=self.name,
        )
        response_stream = request_anthropic_engine(
            config, self.logger, prompt_cache=prompt_cache, stream=True
        )
        
        for chunk in response_stream:
            if chunk and chunk.choices:
                yield {
                    "response": chunk.choices[0].delta.content or "",
                    "usage": {
                        "completion_tokens": 0,  # Token counts only available in final chunk
                        "prompt_tokens": 0,
                        "cache_creation_token": 0,
                        "cache_read_input_tokens": 0,
                    },
                    "finish_reason": chunk.choices[0].finish_reason,
                }

    async def codegen_stream_async(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> AsyncIterator[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        config = create_anthropic_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=1,
            model=self.name,
        )
        response_stream = await request_anthropic_engine_async(
            config, self.logger, prompt_cache=prompt_cache, stream=True
        )
        
        async for chunk in response_stream:
            if chunk and chunk.choices:
                yield {
                    "response": chunk.choices[0].delta.content or "",
                    "usage": {
                        "completion_tokens": 0,  # Token counts only available in final chunk
                        "prompt_tokens": 0,
                        "cache_creation_token": 0,
                        "cache_read_input_tokens": 0,
                    },
                    "finish_reason": chunk.choices[0].finish_reason,
                }

    # specialized codegen with tool
    def codegen_w_tool(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        def _build_response_and_extract(response, messages, iter):
            json_response = {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
            if hasattr(response.choices[0].message, "tool_calls"):
                json_response["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in response.choices[0].message.tool_calls
                ]

            contains_tool = False
            messages.append(json_response)

            response_content = []

            if json_response.get("tool_calls"):
                contains_tool = True
                for tool_call in json_response["tool_calls"]:
                    response_content.append(
                        {
                            "type": "tool_result",
                            "tool_call_id": tool_call["id"],
                            "content": self._USER_REPLY_EDIT_MESSAGE,
                        }
                    )

            if contains_tool:
                messages.append(
                    {
                        "role": "user",
                        "content": response_content,
                    }
                )
            else:
                if iter == 0:
                    # if the first iteration does not contain the tool, likely the model is doing some CoT for debugging
                    # append encouraging message
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please generate editing commands to fix the issue",
                                }
                            ],
                        }
                    )
                    contains_tool = True

            return messages, contains_tool

        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            self.logger.info(f" === Generating ====")
            # initialized the traj
            traj = {
                "response": [],
                "usage": {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "cache_creation_token": 0,
                    "cache_read_input_tokens": 0,
                },
            }

            # create the initial config and messages
            messages = [
                {"role": "user", "content": [{"type": "text", "text": message}]}
            ]

            for iteration in range(self.MAX_CODEGEN_ITERATIONS):
                config = create_anthropic_config(
                    message=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    batch_size=1,
                    model=self.name,
                    tools=self.tools,
                )
                ret = request_anthropic_engine(
                    config,
                    self.logger,
                    prompt_cache=True,  # prompt cache should be always true as we at least should query twice
                )

                if ret:
                    # add the response to the traj
                    # Convert tool calls to dict first if they exist
                    tool_calls_dict = None
                    if hasattr(ret.choices[0].message, "tool_calls"):
                        tool_calls = ret.choices[0].message.tool_calls
                        tool_calls_dict = []
                        for tool_call in tool_calls:
                            tool_dict = {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                            tool_calls_dict.append(tool_dict)
                    
                    # Create response dict after tool calls are converted
                    response_dict = {
                        "role": "assistant",
                        "content": ret.choices[0].message.content,
                    }
                    if tool_calls_dict:
                        response_dict["tool_calls"] = tool_calls_dict
                    
                    traj["response"].append([response_dict])

                    # Log the response
                    self.logger.info(f"Response: {response_dict}")

                    # update the usage
                    traj["usage"]["completion_tokens"] += ret.usage.completion_tokens
                    traj["usage"]["prompt_tokens"] += ret.usage.prompt_tokens
                    traj["usage"]["cache_creation_token"] = 0
                    traj["usage"]["cache_read_input_tokens"] = 0

                    messages, contains_tool = _build_response_and_extract(
                        ret, messages, iteration
                    )

                    if not contains_tool:
                        break
                else:
                    assert (
                        False
                    ), "No response from the engine"  # this should not happen

            if ret:
                trajs.append(traj)
            else:
                trajs.append(self._create_trajectory("", 0, 0))

        return trajs

    def is_direct_completion(self) -> bool:
        return False


class DeepSeekChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def _create_trajectory(self, response, completion_tokens: int, prompt_tokens: int) -> dict:
        """Helper method to create a trajectory object from a response and token counts."""
        return {
            "response": response,
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
            },
        }

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_chatgpt_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_chatgpt_engine(
                config, self.logger, base_url="https://api.deepseek.com"
            )
            if ret:
                trajs.append(self._create_trajectory(
                    ret.choices[0].message.content,
                    ret.usage.completion_tokens,
                    ret.usage.prompt_tokens
                ))
            else:
                trajs.append(self._create_trajectory("", 0, 0))

        return trajs

    async def codegen_async(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_chatgpt_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = await request_chatgpt_engine_async(
                config, self.logger, base_url="https://api.deepseek.com"
            )
            if ret:
                trajs.append(self._create_trajectory(
                    ret.choices[0].message.content,
                    ret.usage.completion_tokens,
                    ret.usage.prompt_tokens
                ))
            else:
                trajs.append(self._create_trajectory("", 0, 0))

        return trajs

    def codegen_stream(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> Iterator[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=1,
            model=self.name,
        )
        response_stream = request_chatgpt_engine(
            config, self.logger, base_url="https://api.deepseek.com", stream=True
        )
        
        for chunk in response_stream:
            if chunk and chunk.choices:
                yield {
                    "response": chunk.choices[0].delta.content or "",
                    "usage": {
                        "completion_tokens": 0,  # Token counts only available in final chunk
                        "prompt_tokens": 0,
                    },
                    "finish_reason": chunk.choices[0].finish_reason,
                }

    async def codegen_stream_async(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> AsyncIterator[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=1,
            model=self.name,
        )
        response_stream = await request_chatgpt_engine_async(
            config, self.logger, base_url="https://api.deepseek.com", stream=True
        )
        
        async for chunk in response_stream:
            if chunk and chunk.choices:
                yield {
                    "response": chunk.choices[0].delta.content or "",
                    "usage": {
                        "completion_tokens": 0,  # Token counts only available in final chunk
                        "prompt_tokens": 0,
                    },
                    "finish_reason": chunk.choices[0].finish_reason,
                }

    def is_direct_completion(self) -> bool:
        return False


def make_model(
    model: str,
    backend: str,
    logger,
    batch_size: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.0,
):
    if backend == "openai":
        return OpenAIChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "anthropic":
        return AnthropicChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "deepseek":
        return DeepSeekChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise NotImplementedError
