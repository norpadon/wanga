import logging
from typing import NamedTuple

from httpx import ConnectError
from ollama import AsyncClient, Client

from ..image_utils import url_to_base64
from ..schema.schema import CallableSchema, JsonSchemaFlavor
from .messages import AssistantMessage, ImageContent, Message, SystemMessage, ToolInvocation, ToolMessage, UserMessage
from .model import (
    FinishReason,
    GenerationParams,
    Model,
    ModelResponse,
    ResponseOption,
    ServiceUnvailableError,
    ToolParams,
    ToolUseMode,
    UsageStats,
)

_logger = logging.getLogger(__name__)


class OllamaModel(Model):
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self._client = Client(host=host)
        self._async_client = AsyncClient(host=host)
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"ollama-{self._model_name}"

    def _list_available_models(self) -> list[str]:
        return [model.model for model in self._client.list()]  # type: ignore

    @property
    def context_length(self) -> int:
        return 8192

    def _get_reply_kwargs(
        self,
        messages: list[Message],
        tools: ToolParams = ToolParams(),
        params: GenerationParams = GenerationParams(),
        num_options: int = 1,
        user_id: str | None = None,
    ) -> dict:
        formatted_messages = _format_messages(messages)
        formatted_tools = _format_tools(tools.tools)
        _check_tool_choice(tools.tool_use_mode)
        if params.force_json:
            format_arg = "json"
        else:
            format_arg = None

        return {
            "model": self._model_name,
            "messages": formatted_messages,
            "tools": formatted_tools,
            "format": format_arg,
            "options": _format_generation_params(params, self.context_length),
        }

    def reply(
        self,
        messages: list[Message],
        tools: ToolParams = ToolParams(),
        params: GenerationParams = GenerationParams(),
        num_options: int = 1,
        user_id: str | None = None,
    ) -> ModelResponse:
        kwargs = self._get_reply_kwargs(messages, tools, params, num_options, user_id)
        try:
            tool_names = [tool.name for tool in tools.tools]
            _logger.debug(f"Ollama request. Messages:\n{messages}\n, Tools:\n{tool_names}")
            response = self._client.chat(**kwargs)
            result = _parse_response(response)
            _logger.debug("Ollama Response: %s", result)
            return result
        except ConnectError as e:
            raise ServiceUnvailableError(e) from e

    async def reply_async(
        self,
        messages: list[Message],
        tools: ToolParams = ToolParams(),
        params: GenerationParams = GenerationParams(),
        num_options: int = 1,
        user_id: str | None = None,
    ) -> ModelResponse:
        kwargs = self._get_reply_kwargs(messages, tools, params, num_options, user_id)
        try:
            tool_names = [tool.name for tool in tools.tools]
            _logger.debug(f"Ollama request.\nMessages:\n{messages}\nTools:\n{tool_names}")
            response = await self._async_client.chat(**kwargs)
            result = _parse_response(response)
            _logger.debug(f"Ollama Response: {result}")
            return result
        except ConnectError as e:
            raise ServiceUnvailableError(e) from e


def _format_generation_params(params: GenerationParams, context_length: int) -> dict:
    result = {}
    result["num_ctx"] = context_length
    if params.max_tokens is not None:
        result["num_predict"] = params.max_tokens
    if params.temperature is not None:
        result["temperature"] = params.temperature
    if params.top_p is not None:
        result["top_p"] = params.top_p
    if params.frequency_penalty is not None:
        result["repeat_penalty"] = params.frequency_penalty
    if params.stop_sequences is not None:
        result["stop"] = params.stop_sequences
    if params.random_seed is not None:
        result["seed"] = params.random_seed

    return result


def _format_messages(messages: list[Message]) -> list[dict]:
    return [_format_message(message) for message in messages]


def _format_tool(tool: CallableSchema) -> dict:
    function_schema = tool.json_schema(flavor=JsonSchemaFlavor.OPENAI, include_long_description=True)
    return dict(function=function_schema)


def _format_tools(tools: list[CallableSchema]) -> list[dict]:
    return [_format_tool(tool) for tool in tools]


def _check_tool_choice(tool_use_mode: ToolUseMode | str) -> None:
    pass
    # if tool_use_mode != ToolUseMode.AUTO:
    #    raise ValueError(f"Unsupported tool use mode: {tool_use_mode}")


def _format_image_content(image: ImageContent) -> bytes:
    if image.base64 is not None:
        result = image.base64
    elif image.url is not None:
        result = url_to_base64(image.url)
    else:
        raise ValueError("ImageContent must have either a URL or base64 data")
    return result.encode()


class MessageContent(NamedTuple):
    text: str
    images: list[str]


def _format_content(content: str | list[str | ImageContent]) -> MessageContent:
    if isinstance(content, str):
        return MessageContent(text=content, images=[])

    text = []
    images = []
    for item in content:
        if isinstance(item, ImageContent):
            images.append(_format_image_content(item))
        else:
            text.append(item)

    text = "".join(text)
    return MessageContent(text=text, images=images)


def _format_functon_call(function_call: ToolInvocation) -> dict:
    if not isinstance(function_call.arguments, dict):
        raise ValueError(f"Ollama API doesn't support non-dict function call arguments: {function_call}")
    return dict(
        function=dict(
            name=function_call.name,
            arguments=function_call.arguments,
        ),
    )


def _format_message(message: Message) -> dict:
    match message:
        case UserMessage(content=content, name=name):
            if name:
                raise ValueError("Ollama doesn't support named message roles.")
            text_content, image_content = _format_content(content)
            result = dict(
                role="user",
                content=text_content,
                images=image_content or None,
            )
        case AssistantMessage(content=content, name=name, tool_invocations=tool_invocations):
            if name:
                raise ValueError("Ollama doesn't support named message roles.")
            tool_calls = [_format_functon_call(call) for call in tool_invocations]
            if content is None:
                text_content, image_content = "", []
            else:
                text_content, image_content = _format_content(content)
            return dict(
                role="assistant",
                content=text_content,
                images=image_content or None,
                tool_calls=tool_calls,
            )
        case SystemMessage(content=content, name=name):
            if name:
                raise ValueError("Ollama doesn't support named message roles.")
            text_content, image_content = _format_content(content)
            return dict(
                role="system",
                content=text_content,
                images=image_content or None,
            )
        case ToolMessage(content=content, invocation_id=_):
            text_content, image_content = _format_content(content)
            return dict(
                role="tool",
                content=text_content,
                images=image_content or None,
            )
        case _:
            raise ValueError(f"Unknown message type: {message}")
    return result


def _parse_tool_invocation(tool: dict) -> ToolInvocation:
    return ToolInvocation(invocation_id=None, name=tool["function"]["name"], arguments=tool["function"]["arguments"])  # type: ignore


def _parse_response(response: dict) -> ModelResponse:
    tool_invocations = [_parse_tool_invocation(tool) for tool in response["message"]["tool_calls"] or []]
    if tool_invocations:
        finish_reason = FinishReason.TOOL_CALL
    else:
        finish_reason = FinishReason.STOP
    option = ResponseOption(
        message=AssistantMessage(
            content=response["message"]["content"],
            tool_invocations=tool_invocations,
        ),
        finish_reason=finish_reason,
    )
    usage = UsageStats(
        prompt_tokens=response["prompt_eval_count"] or 0,
        response_tokens=response["eval_count"] or 0,
    )
    return ModelResponse(response_options=[option], usage=usage)
