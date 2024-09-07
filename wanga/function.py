import inspect
import logging
from collections.abc import Iterable
from functools import wraps
from typing import Callable, Generic, ParamSpec, TypeVar

from attrs import field, frozen
from jinja2 import Template

from .models import GenerationParams
from .schema import DEFAULT_SCHEMA_EXTRACTOR, CallableSchema, SchemaExtractor
from .schema.schema import ObjectField, ObjectNode, PrimitiveNode, SchemaNode, UndefinedNode
from .templates import get_local_variables, make_template

_logger = logging.getLogger(__name__)


__all__ = ["ai_function"]


_P = ParamSpec("_P")
_R = TypeVar("_R")


_RESPONSE_TOOL_NAME = "submit_response"
_RESPONSE_FIELD_NAME = "response"
_RESPONSE_TOOL_PROMPT = "Call this function to respond to the user."


@frozen
class AIFunction(Generic[_P, _R]):
    signature: inspect.Signature
    prompt_template: Template
    return_schema: CallableSchema | None
    tools: list[CallableSchema] = field(factory=list)
    preferred_models: list[str] = field(factory=list)
    generation_params: GenerationParams = GenerationParams()
    max_retries_on_invalid_output: int = 3


def _extract_return_schema(callable: Callable, extractor: SchemaExtractor) -> CallableSchema | None:
    callable_schema = extractor.extract_schema(callable)
    if isinstance(callable_schema.return_schema, UndefinedNode):
        raise ValueError("Function must have a concrete return type annotation.")
    return _make_response_schema(callable_schema.return_schema)


def _extract_promt_template(callable: Callable) -> Template:
    docstring = inspect.getdoc(callable)
    parameters = inspect.signature(callable).parameters
    if docstring is None:
        raise ValueError("Prompt is missing.")
    local_variables = get_local_variables(docstring)
    for variable_name in local_variables:
        if variable_name not in parameters:
            raise ValueError(f"Variable {variable_name} is not a parameter of the function.")
    for parameter in parameters.values():
        if parameter.name not in local_variables:
            _logger.warning(f"Parameter {parameter.name} is not used in the prompt.")
    return make_template(docstring)


def _extract_ai_function(
    callable: Callable[_P, _R],
    tools: list[Callable],
    preferred_models: list[str],
    generation_params: GenerationParams,
    schema_extractor: SchemaExtractor,
    max_retries_on_invalid_output: int,
) -> AIFunction[_P, _R]:
    return_schema = _extract_return_schema(callable, schema_extractor)
    prompt_template = _extract_promt_template(callable)
    tools_schemas = [schema_extractor.extract_schema(tool) for tool in tools]
    return AIFunction(
        signature=inspect.signature(callable),
        prompt_template=prompt_template,
        return_schema=return_schema,
        tools=tools_schemas,
        preferred_models=preferred_models,
        generation_params=generation_params,
        max_retries_on_invalid_output=max_retries_on_invalid_output,
    )


def _make_wrapper(ai_function: AIFunction[_P, _R], wrap_as: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(wrap_as)
    def function_object_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        from .runtime import get_runtime

        return get_runtime().execute(ai_function, *args, **kwargs)

    function_object_wrapper.__ai_function = ai_function  # type: ignore
    return function_object_wrapper


def _make_response_schema(response_node: SchemaNode) -> CallableSchema | None:
    # If the function returns a string, we don't need to define a response schema.
    if response_node == PrimitiveNode(str):
        return None

    def reply(response):
        return response

    call_schema = ObjectNode(
        reply,
        inspect.signature(reply),
        name=_RESPONSE_TOOL_NAME,
        fields=[
            ObjectField(_RESPONSE_FIELD_NAME, response_node, hint=None, required=True),
        ],
        hint=_RESPONSE_TOOL_PROMPT,
    )

    return CallableSchema(call_schema, call_schema, None)


def ai_function(
    tools: Iterable[Callable] | None = None,
    preferred_models: str | Iterable[str] | None = None,
    generation_params: GenerationParams = GenerationParams(),
    schema_extractor: SchemaExtractor | None = None,
    max_retries_on_invalid_output: int = 3,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    tools = tools or []
    preferred_models = preferred_models or []
    if isinstance(preferred_models, str):
        preferred_models = [preferred_models]
    schema_extractor = schema_extractor or DEFAULT_SCHEMA_EXTRACTOR

    def decorator(callable: Callable[_P, _R]) -> Callable[_P, _R]:
        ai_function = _extract_ai_function(
            callable,
            list(tools),
            list(preferred_models),
            generation_params,
            schema_extractor,
            max_retries_on_invalid_output,
        )
        return _make_wrapper(ai_function, callable)

    return decorator
