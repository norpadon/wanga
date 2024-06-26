import inspect
from collections.abc import Callable
from types import NoneType, UnionType
from typing import Any, get_args, get_origin

from attrs import frozen
from docstring_parser import parse as parse_docstring

from .normalize import normalize_annotation
from .schema import (
    CallableSchema,
    MappingNode,
    ObjectField,
    ObjectNode,
    PrimitiveNode,
    SchemaNode,
    SequenceNode,
    TupleNode,
    UndefinedNode,
    UnionNode,
)

__all__ = [
    "annotation_to_schema",
    "extract_schema",
]


@frozen
class DocstringHints:
    object_hint: str | None
    param_to_hint: dict[str, str]


def extract_hints(callable: Callable) -> DocstringHints:
    docstring = inspect.getdoc(callable)
    if docstring is not None:
        docstring = parse_docstring(docstring)
        object_hint = docstring.short_description
        param_to_hint = {
            param.arg_name: param.description
            for param in docstring.params
            if param.description
        }
    else:
        object_hint = None
        param_to_hint = {}

    if isinstance(callable, type) and hasattr(callable, "__init__"):
        init_hints = extract_hints(callable.__init__)
        object_hint = object_hint or init_hints.object_hint
        param_to_hint.update(init_hints.param_to_hint)

    return DocstringHints(object_hint, param_to_hint)


def annotation_to_schema(annotation) -> SchemaNode:
    r"""Extract a schema from the type annotation."""
    if annotation in [Any, None]:
        return UndefinedNode(original_annotation=annotation)

    annotation = normalize_annotation(annotation, concretize=True)

    if annotation in [int, float, str, bool]:
        return PrimitiveNode(primitive_type=annotation)  # type: ignore

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        # Object node, so we need to extract fields from the constructor signature.
        init_schema = extract_schema(annotation)
        return init_schema.call_schema

    # Normalization step has already converted all abstract classes to the corresponding concrete types.
    # So we can safely check against list and dict.
    if issubclass(origin, list):
        assert (
            len(args) == 1
        ), "Sequence type annotation should have exactly one argument."
        return SequenceNode(
            sequence_type=origin,
            item_schema=annotation_to_schema(args[0]),
        )
    if issubclass(origin, tuple):
        return TupleNode(
            tuple_type=origin,
            item_schemas=[annotation_to_schema(arg) for arg in args],
        )
    if issubclass(origin, dict):
        assert (
            len(args) == 2
        ), "Mapping type annotation should have exactly two arguments."
        return MappingNode(
            mapping_type=origin,
            key_schema=annotation_to_schema(args[0]),
            value_schema=annotation_to_schema(args[1]),
        )
    if issubclass(origin, UnionType):
        arg_schemas = []
        for arg in args:
            if arg is NoneType:
                arg_schemas.append(None)
            else:
                arg_schemas.append(annotation_to_schema(arg))
        return UnionNode(options=arg_schemas)

    raise ValueError(f"Unsupported type annotation: {annotation}")


def extract_schema(callable: Callable) -> CallableSchema:
    r"""Extract a schema from a callable."""
    signature = inspect.signature(callable, eval_str=True)
    return_type = signature.return_annotation

    hints = extract_hints(callable)

    if return_type is inspect.Signature.empty:
        return_schema = UndefinedNode(original_annotation=Any)
    else:
        return_schema = annotation_to_schema(return_type)

    object_fields = []
    for param in signature.parameters.values():
        if param.annotation is inspect.Signature.empty:
            arg_schema = UndefinedNode(original_annotation=Any)
        else:
            arg_schema = annotation_to_schema(param.annotation)
        field = ObjectField(
            param.name,
            arg_schema,
            hints.param_to_hint.get(param.name),
            param.default is param.empty,
        )
        object_fields.append(field)

    return CallableSchema(
        call_schema=ObjectNode(
            name=callable.__name__,
            fields=object_fields,
            hint=hints.object_hint,
        ),
        return_schema=return_schema,
    )
