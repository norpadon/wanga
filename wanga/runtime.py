from collections.abc import Iterable
from contextvars import ContextVar
from typing import Any, NamedTuple, ParamSpec, TypeVar

from attr import frozen
from jinja2 import Template

from .function import _RESPONSE_TOOL_NAME, AIFunction
from .models import AssistantMessage, Message, Model, ToolMessage, parse_messages
from .models.model import FinishReason, GenerationParams, ToolParams, ToolUseMode
from .schema import SchemaValidationError

__all__ = ["Runtime"]


_P = ParamSpec("_P")
_R = TypeVar("_R")


class WangaRuntimeError(Exception):
    pass


class ContentFilterError(ValueError, WangaRuntimeError):
    pass


class OutputTooLongError(RuntimeError, WangaRuntimeError):
    pass


def render_prompt(template: Template, arguments: dict[str, Any]) -> list[Message]:
    message_str = template.render(arguments)
    return parse_messages(message_str)


class ToolInvocationResults(NamedTuple):
    messages: list[ToolMessage]
    errors: list[Exception]


@frozen
class FinalResponse:
    value: Any


def invoke_tools(message: AssistantMessage, tool_params: ToolParams) -> ToolInvocationResults | FinalResponse:
    messages = []
    errors = []
    for invocation in message.tool_invocations:
        tool = tool_params.get_tool(invocation.name)
        try:
            tool_output = tool.eval(invocation.arguments)
            if tool.name == _RESPONSE_TOOL_NAME:
                return FinalResponse(tool_output)
        except SchemaValidationError as e:
            tool_output = str(e)
            errors.append(e)
        messages.append(ToolMessage(invocation.invocation_id, str(tool_output)))
    return ToolInvocationResults(messages, errors)


def call_and_use_tools(
    model: Model,
    messages: list[Message],
    tool_params: ToolParams,
    generation_params: GenerationParams,
    allow_plain_text_response: bool,
    max_retries_on_invalid_output: int,
) -> Any:
    messages = list(messages)
    retries_left = max_retries_on_invalid_output
    while True:
        response = model.reply(messages, tool_params, generation_params).response_options[0]
        message = response.message
        messages.append(message)
        if response.finish_reason == FinishReason.STOP:
            if allow_plain_text_response:
                return message.content
        if response.finish_reason in [FinishReason.STOP, FinishReason.TOOL_CALL]:
            if message.tool_invocations:
                tool_invocation_results = invoke_tools(message, tool_params)
                if isinstance(tool_invocation_results, FinalResponse):
                    return tool_invocation_results.value
                tool_messages, errors = tool_invocation_results
                messages.extend(tool_messages)
                if errors:
                    if retries_left == 0:
                        raise errors[0]
                    retries_left -= 1
                else:
                    retries_left = max_retries_on_invalid_output
            else:
                raise RuntimeError("Model returned a stop response without any tool invocations.")
        elif response.finish_reason == FinishReason.CONTENT_FILTER:
            raise ContentFilterError(f"Content filter triggered: {message.content}")
        elif FinishReason.LENGTH:
            raise OutputTooLongError(message.content)


class Runtime:
    def _get_model(self, model: str | Model) -> Model:
        if isinstance(model, Model):
            return model
        else:
            raise NotImplementedError

    def __init__(self, model: str | Model | Iterable[str | Model]):
        if not isinstance(model, Iterable):
            model = [model]
        self.models = [self._get_model(m) for m in model]

    def __enter__(self):
        self._token = _global_runtime.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _global_runtime.reset(self._token)

    def _select_model(self, preferred_models: list[str]) -> Model:
        name_to_model = {model.name: model for model in self.models}
        for preferred_model in preferred_models:
            model = name_to_model.get(preferred_model)
            if model is not None:
                return model
        return self.models[0]

    def execute(self, ai_function: AIFunction[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
        bound_arguments = ai_function.signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        prompt = render_prompt(ai_function.prompt_template, bound_arguments.arguments)
        generation_params = ai_function.generation_params
        model = self._select_model(ai_function.preferred_models)

        tools = list(ai_function.tools)
        if ai_function.return_schema is not None:
            tools.append(ai_function.return_schema)
            tool_mode = ToolUseMode.FORCE
        else:
            tool_mode = ToolUseMode.AUTO
        tool_params = ToolParams(tools, tool_mode)

        return call_and_use_tools(
            model,
            prompt,
            tool_params,
            generation_params,
            ai_function.return_schema is None,
            ai_function.max_retries_on_invalid_output,
        )


def get_runtime() -> Runtime:
    result = _global_runtime.get()
    if result is None:
        raise ValueError("Wanga Runtime is not initialized.")
    return result


_global_runtime: ContextVar[Runtime | None] = ContextVar("global_runtime", default=None)
