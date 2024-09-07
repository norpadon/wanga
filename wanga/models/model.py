from enum import Enum

from attrs import field, frozen

from ..schema.schema import CallableSchema
from .messages import AssistantMessage, Message

__all__ = [
    "ToolUseMode",
    "ToolParams",
    "GenerationParams",
    "UsageStats",
    "FinishReason",
    "ResponseOption",
    "ModelResponse",
    "ModelError",
    "AuthenticationError",
    "PromptError",
    "ModelTimeoutError",
    "InvalidJsonError",
    "RateLimitError",
    "ServiceUnvailableError",
    "Model",
]


class ToolUseMode(Enum):
    AUTO = "auto"
    FORCE = "force"
    NEVER = "never"


@frozen
class ToolParams:
    tools: list[CallableSchema] = field(factory=list)
    tool_use_mode: ToolUseMode | str = ToolUseMode.AUTO
    allow_parallel_calls: bool = False

    def get_tool(self, tool_name: str) -> CallableSchema:
        name_to_tool = {tool.name: tool for tool in self.tools}
        return name_to_tool[tool_name]


@frozen
class GenerationParams:
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop_sequences: list[str] | None = None
    random_seed: int | None = None
    force_json: bool = False


@frozen
class UsageStats:
    prompt_tokens: int
    response_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens


class FinishReason(Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"


@frozen
class ResponseOption:
    message: AssistantMessage
    finish_reason: FinishReason


@frozen
class ModelResponse:
    response_options: list[ResponseOption]
    usage: UsageStats


class ModelError(Exception):
    r"""Base class for all exceptions raised by models."""


class AuthenticationError(ModelError):
    r"""Raised when the API credentials are invalid."""


class PromptError(ValueError, ModelError):
    r"""Raised when the prompt is malformed or uses unsupported features."""


class ModelTimeoutError(TimeoutError, ModelError):
    r"""Raised when the model takes too long to generate a response."""


class InvalidJsonError(RuntimeError, ModelError):
    r"""Raised when the model returns a malformed JSON as a response to a function call."""


class RateLimitError(ModelError):
    r"""Raised when the request limit is exceeded."""


class ServiceUnvailableError(RuntimeError, ModelError):
    r"""Raised when the API is down."""


class Model:
    def reply(
        self,
        messages: list[Message],
        tools: ToolParams = ToolParams(),
        params: GenerationParams = GenerationParams(),
        num_options: int = 1,
        user_id: str | None = None,
    ) -> ModelResponse:
        raise NotImplementedError

    async def reply_async(
        self,
        messages: list[Message],
        tools: ToolParams = ToolParams(),
        params: GenerationParams = GenerationParams(),
        num_options: int = 1,
        user_id: str | None = None,
    ) -> ModelResponse:
        raise NotImplementedError

    @property
    def name(self) -> str:
        r"""Name of the model in the format {provider_name}-{model_name}.

        Example: "openai-gpt-3.5-turbo"
        """
        raise NotImplementedError

    @property
    def context_length(self) -> int:
        r"""Maximum number of tokens that can be used in the prompt and completion."""
        raise NotImplementedError

    def estimate_num_tokens(self, messages: list[Message], tools: ToolParams) -> int:
        r"""Returns the rough estimate of the total number of tokens in the prompt.

        May return inaccurate results, use `calculate_num_tokens` instead for precise results.
        Note that `calculate_num_tokens` may send request to the LLM api, which will count towards your usage bill.
        """
        raise NotImplementedError

    def calculate_num_tokens(self, messages: list[Message], tools: ToolParams) -> int:
        r"""Returns the precise number of tokens in the prompt.

        This method will send a request to the LLM api, which will count towards your usage bill.
        """
        raise NotImplementedError
