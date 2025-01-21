import json
import re

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..common import openai, openai_function_tokens
from ..schema import CallableSchema, JsonSchemaFlavor
from .messages import AssistantMessage, ImageContent, Message, SystemMessage, ToolInvocation, ToolMessage, UserMessage
from .model import (
    AuthenticationError,
    FinishReason,
    GenerationParams,
    InvalidJsonError,
    Model,
    ModelResponse,
    ModelTimeoutError,
    PromptError,
    RateLimitError,
    ResponseOption,
    ServiceUnvailableError,
    ToolParams,
    ToolUseMode,
    UsageStats,
)

_TOO_MANY_TOKENS = 100_000_000_000

_NUM_TOKENS_ERR_RE = re.compile(r"\((?P<messages>\d+) in the messages(, (?P<functions>\d+) in the functions,)?")


__all__ = ["OpenAIModel"]


class OpenAIModel(Model):
    _NAME_PREFIX_TO_CONTEXT_LENGTH = {
        "gpt-3.5-turbo": 16538,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
    }
    # We sort the keys such that 'gpt-4-turbo' comes before 'gpt-4'.
    _NAME_PREFIX_TO_CONTEXT_LENGTH = {k: v for k, v in sorted(_NAME_PREFIX_TO_CONTEXT_LENGTH.items(), reverse=True)}

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        request_timeout: float = 10 * 60,
        num_retries: int = 0,
        retry_on_request_limit: bool = True,
    ):
        self._retry_on_request_limit = retry_on_request_limit
        self._num_retries = num_retries
        self._client = openai.OpenAI(api_key=api_key, timeout=request_timeout, max_retries=0)
        self._async_client = openai.AsyncOpenAI(api_key=api_key, timeout=request_timeout, max_retries=0)
        if model_name not in self._list_available_models():
            raise ValueError(f"Model {model_name} is not available")
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"openai-{self._model_name}"

    def _list_available_models(self) -> list[str]:
        model_list = self._client.models.list()
        return [model.id for model in model_list]

    @property
    def context_length(self) -> int:
        for key, value in self._NAME_PREFIX_TO_CONTEXT_LENGTH.items():
            if self._model_name.startswith(key):
                return value
        raise ValueError(f"Unknown model name: {self._model_name}")

    def estimate_num_tokens(self, messages: list[Message], tools: ToolParams) -> int:
        formatted_messages = _format_messages(messages)
        if tools.tools:
            formatted_tools = [d["function"] for d in _format_tools(tools.tools)]
        else:
            formatted_tools = None
        if tools.tool_use_mode:
            function_call = _format_tool_choice(tools.tool_use_mode)
        else:
            function_call = None
        return openai_function_tokens.estimate_tokens(
            formatted_messages,
            formatted_tools,  # type: ignore
            function_call,
        )

    def calculate_num_tokens(self, messages: list[Message], tools: ToolParams) -> int:
        try:
            self.reply(messages, tools, GenerationParams(max_tokens=_TOO_MANY_TOKENS))
        except PromptError as e:
            err_string = str(e)
            match = _NUM_TOKENS_ERR_RE.search(err_string)
            if match is None:
                raise RuntimeError(f"Failed to parse the error message: {err_string}")
            result = int(match.group("messages"))
            if match.group("functions"):
                result += int(match.group("functions"))
            return result
        assert False

    def _get_reply_kwargs(
        self,
        messages: list[Message],
        tools: ToolParams = ToolParams(),
        params: GenerationParams = GenerationParams(),
        num_options: int = 1,
        user_id: str | None = None,
    ) -> dict:
        formatted_messages = _format_messages(messages)
        if tools.tools:
            formatted_tools = _format_tools(tools.tools)
            tool_choice = _format_tool_choice(tools.tool_use_mode)
        else:
            formatted_tools = None
            tool_choice = None
        result = dict(
            model=self._model_name,
            messages=formatted_messages,
            frequency_penalty=params.frequency_penalty,
            max_tokens=params.max_tokens,
            n=num_options,
            presence_penalty=params.presence_penalty,
            response_format={"type": "json_object"} if params.force_json else None,
            seed=params.random_seed,
            stop=params.stop_sequences,
            temperature=params.temperature,
            top_p=params.top_p,
            tools=formatted_tools,
            tool_choice=tool_choice,
            user=user_id,
        )
        # OpenAI API breaks if we explicitly pass default values for some keys.
        result = {k: v for k, v in result.items() if v is not None}
        if tools.tools:
            result["parallel_tool_calls"] = tools.allow_parallel_calls
        return result

    def reply(
        self,
        messages: list[Message],
        tools: ToolParams = ToolParams(),
        params: GenerationParams = GenerationParams(),
        num_options: int = 1,
        user_id: str | None = None,
    ) -> ModelResponse:
        kwargs = self._get_reply_kwargs(messages, tools, params, num_options, user_id)

        @self._create_retry_decorator()
        def _reply_with_retry():
            try:
                response = self._client.chat.completions.create(**kwargs)
                return _parse_response(response)
            except openai.OpenAIError as e:
                raise _wrap_error(e)

        return _reply_with_retry()

    async def reply_async(
        self,
        messages: list[Message],
        tools: ToolParams = ToolParams(),
        params: GenerationParams = GenerationParams(),
        num_options: int = 1,
        user_id: str | None = None,
    ) -> ModelResponse:
        kwargs = self._get_reply_kwargs(messages, tools, params, num_options, user_id)

        @self._create_retry_decorator()
        async def _reply_async_with_retry():
            try:
                response = await self._async_client.chat.completions.create(**kwargs)
                return _parse_response(response)
            except openai.OpenAIError as e:
                raise _wrap_error(e)

        return await _reply_async_with_retry()

    def _create_retry_decorator(self):
        retry_conditions = retry_if_exception_type(ServiceUnvailableError)
        if self._retry_on_request_limit:
            retry_conditions |= retry_if_exception_type(RateLimitError)

        return retry(
            stop=stop_after_attempt(self._num_retries + 1),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_conditions,
            reraise=True,
        )


def _wrap_error(error: Exception) -> Exception:
    match error:
        case openai.APITimeoutError() as e:
            return ModelTimeoutError(e)
        case openai.BadRequestError() as e:
            return PromptError(e)
        case openai.AuthenticationError() as e:
            return AuthenticationError(e)
        case openai.RateLimitError() as e:
            return RateLimitError(e)
        case openai.InternalServerError() | openai.APIConnectionError() as e:
            return ServiceUnvailableError(e)
        case openai.APIError() as e:
            # Handle other API errors that might be retryable
            return ServiceUnvailableError(e)
        case _:
            return error


def _format_messages(messages: list[Message]) -> list[dict]:
    return [_format_message(message) for message in messages]


def _format_tool(tool: CallableSchema) -> dict:
    return {
        "type": "function",
        "function": tool.json_schema(flavor=JsonSchemaFlavor.OPENAI, include_long_description=True),
    }


def _format_tools(tools: list[CallableSchema]) -> list:
    return [_format_tool(tool) for tool in tools]


def _format_tool_choice(tool_use_mode: ToolUseMode | str):
    match tool_use_mode:
        case ToolUseMode.AUTO:
            return "auto"
        case ToolUseMode.FORCE:
            return "required"
        case ToolUseMode.NEVER:
            return "none"
        case _:
            return {"type": "function", "name": tool_use_mode}


def _format_image_content(image: ImageContent) -> dict:
    if image.url is not None:
        url = image.url
    else:
        url = f"data:image/jpeg;base64,{image.base64}"
    return {
        "type": "image_url",
        "image_url": {
            "url": url,
            "detail": str(image.detail),
        },
    }


def _format_content(content: str | list[str | ImageContent]):
    if isinstance(content, str):
        return content
    result = []
    for item in content:
        if isinstance(item, ImageContent):
            result.append(_format_image_content(item))
        else:
            result.append({"type": "text", "text": item})
    return result


def _format_functon_call(function_call: ToolInvocation):
    formatted_json = json.dumps(function_call.arguments, sort_keys=True)
    return {
        "id": function_call.invocation_id,
        "type": "function",
        "function": {
            "name": function_call.name,
            "arguments": formatted_json,
        },
    }


def _format_message(message: Message) -> dict:
    result: dict
    match message:
        case UserMessage(content=content, name=name):
            result = {"role": "user", "content": _format_content(content)}
            if name:
                result["name"] = name
        case AssistantMessage(content=content, name=name, tool_invocations=tool_invocations):
            result = {"role": "assistant"}
            if content:
                result["content"] = content
            if name:
                result["name"] = name
            if tool_invocations:
                result["tool_calls"] = [_format_functon_call(call) for call in tool_invocations]
        case SystemMessage(content=content, name=name):
            result = {"role": "system", "content": content}
            if name:
                result["name"] = name
        case ToolMessage(content=content, invocation_id=invocation_id):
            if not isinstance(content, str):
                raise ValueError("OpenAI doesn't support images in tool messages.")
            return {"role": "tool", "content": content, "tool_call_id": invocation_id}
        case _:
            raise ValueError(f"Unknown message type: {message}")
    return result


def _parse_usage(usage) -> UsageStats:
    return UsageStats(
        prompt_tokens=usage.prompt_tokens,
        response_tokens=usage.completion_tokens,
    )


def _parse_finish_reason(reason: str) -> FinishReason:
    match reason:
        case "stop":
            return FinishReason.STOP
        case "length":
            return FinishReason.LENGTH
        case "tool_calls":
            return FinishReason.TOOL_CALL
        case "content_filter":
            return FinishReason.CONTENT_FILTER
        case _:
            raise ValueError(f"Unknown finish reason: {reason}")


def _parse_tool_invocation(tool) -> ToolInvocation:
    try:
        arguments = json.loads(tool.function.arguments)
    except json.JSONDecodeError as e:
        raise InvalidJsonError(f"Failed to parse tool invocation arguments: {tool}") from e
    return ToolInvocation(invocation_id=tool.id, name=tool.function.name, arguments=arguments)


def _parse_response(response) -> ModelResponse:
    usage = _parse_usage(response.usage)
    choices = sorted(response.choices, key=lambda choice: choice.index)
    response_options = []
    for choice in choices:
        message = choice.message
        tool_invocations = [_parse_tool_invocation(tool) for tool in message.tool_calls or []]
        option = ResponseOption(
            message=AssistantMessage(
                content=message.content,
                tool_invocations=tool_invocations,
            ),
            finish_reason=_parse_finish_reason(choice.finish_reason),
        )
        response_options.append(option)

    return ModelResponse(response_options=response_options, usage=usage)
