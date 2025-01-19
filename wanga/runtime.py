from collections.abc import Iterable
from contextlib import AsyncContextDecorator, ContextDecorator
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, AsyncContextManager, ContextManager, NamedTuple, ParamSpec, TypeVar

from attr import frozen
from jinja2 import Template

from .function import _RESPONSE_TOOL_NAME, AIFunction
from .models import AssistantMessage, Message, Model, ToolMessage, parse_messages
from .models.model import FinishReason, GenerationParams, ResponseOption, ToolParams, ToolUseMode
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


@dataclass
class InvocationState:
    messages: list[Message]
    tool_params: ToolParams
    generation_params: GenerationParams
    allow_plain_text_response: bool
    max_retries_on_invalid_output: int
    result: Any | None = None


def _handle_model_response(response: ResponseOption, state: InvocationState) -> InvocationState:
    message = response.message

    new_messages = list(state.messages)
    new_messages.append(message)
    state = replace(state, messages=new_messages)

    if response.finish_reason == FinishReason.STOP:
        if state.allow_plain_text_response:
            return replace(state, result=message.content)
    if response.finish_reason in [FinishReason.STOP, FinishReason.TOOL_CALL]:
        if message.tool_invocations:
            tool_invocation_results = invoke_tools(message, state.tool_params)
            if isinstance(tool_invocation_results, FinalResponse):
                return replace(state, result=tool_invocation_results.value)
            tool_messages, errors = tool_invocation_results
            new_messages.extend(tool_messages)
            state = replace(state, messages=new_messages)
            if errors:
                if state.max_retries_on_invalid_output == 0:
                    raise errors[0]
                state = replace(state, max_retries_on_invalid_output=state.max_retries_on_invalid_output - 1)
        else:
            raise RuntimeError("Model returned a stop response without any tool invocations.")
    elif response.finish_reason == FinishReason.CONTENT_FILTER:
        raise ContentFilterError(f"Content filter triggered: {message.content}")
    elif FinishReason.LENGTH:
        raise OutputTooLongError(message.content)

    return state


def call_and_use_tools(
    model: Model,
    messages: list[Message],
    tool_params: ToolParams,
    generation_params: GenerationParams,
    allow_plain_text_response: bool,
    max_retries_on_invalid_output: int,
) -> Any:
    state = InvocationState(
        messages,
        tool_params,
        generation_params,
        allow_plain_text_response,
        max_retries_on_invalid_output,
    )
    while True:
        response = model.reply(state.messages, state.tool_params, state.generation_params).response_options[0]
        state = _handle_model_response(response, state)
        if state.result is not None:
            return state.result


async def call_and_use_tools_async(
    model: Model,
    messages: list[Message],
    tool_params: ToolParams,
    generation_params: GenerationParams,
    allow_plain_text_response: bool,
    max_retries_on_invalid_output: int,
) -> Any:
    state = InvocationState(
        messages,
        tool_params,
        generation_params,
        allow_plain_text_response,
        max_retries_on_invalid_output,
    )
    while True:
        response = (
            await model.reply_async(state.messages, state.tool_params, state.generation_params)
        ).response_options[0]
        state = _handle_model_response(response, state)
        if state.result is not None:
            return state.result


class Runtime(ContextDecorator, AsyncContextDecorator, ContextManager, AsyncContextManager):
    def _get_model(self, model: str | Model) -> Model:
        if isinstance(model, Model):
            return model
        else:
            raise NotImplementedError

    def __init__(self, model: str | Model | Iterable[str | Model]):
        if not isinstance(model, Iterable):
            model = [model]
        self.models = [self._get_model(m) for m in model]

    async def __aenter__(self):
        self._token = _global_runtime.set(self)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        _global_runtime.reset(self._token)

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

    def _prepare_kwargs(self, ai_function: AIFunction[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> dict[str, Any]:
        bound_arguments = ai_function.signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        messages = render_prompt(ai_function.prompt_template, bound_arguments.arguments)
        generation_params = ai_function.generation_params
        model = self._select_model(ai_function.preferred_models)

        tools = list(ai_function.tools)
        if ai_function.return_schema is not None:
            tools.append(ai_function.return_schema)
            tool_mode = ToolUseMode.FORCE
        else:
            tool_mode = ToolUseMode.AUTO
        tool_params = ToolParams(tools, tool_mode)

        return dict(
            model=model,
            messages=messages,
            tool_params=tool_params,
            generation_params=generation_params,
            allow_plain_text_response=ai_function.return_schema is None,
            max_retries_on_invalid_output=ai_function.max_retries_on_invalid_output,
        )

    def execute(self, ai_function: AIFunction[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
        call_kwargs = self._prepare_kwargs(ai_function, *args, **kwargs)
        return call_and_use_tools(**call_kwargs)

    async def execute_async(self, ai_function: AIFunction[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
        call_kwargs = self._prepare_kwargs(ai_function, *args, **kwargs)
        return await call_and_use_tools_async(**call_kwargs)


def get_runtime() -> Runtime:
    result = _global_runtime.get()
    if result is None:
        raise ValueError("Wanga Runtime is not initialized.")
    return result


_global_runtime: ContextVar[Runtime | None] = ContextVar("global_runtime", default=None)
