import inspect
from collections.abc import Callable
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

from attrs import define, field, frozen
from docstring_parser import parse as parse_docstring

from .extractor_fns import ExtractorFn, extract_datetime
from .normalize import normalize_annotation
from .schema import (
    CallableSchema,
    LiteralNode,
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
from .utils import strip_self

__all__ = [
    "SchemaExtractor",
    "DEFAULT_SCHEMA_EXTRACTOR",
]


@frozen
class DocstringHints:
    object_hint: str | None
    long_decsription: str | None
    param_to_hint: dict[str, str]


class SchemaExtractionError(Exception):
    pass


@define
class SchemaExtractor:
    r"""Extracts schemas from callables.

    Should never be initialized directly by the end user, instead, use `wanga.schema.default_schema_extractor`.
    If you want to add new extraction functions, use the `register_extract_fn` method.

    Attributes:
        extractor_functions: List of functions that take type annotation as an input
            and try to produce the `CallableSchema`.
    """

    exctractor_fns: list[ExtractorFn] = field(factory=list)

    def __attrs_post_init__(self) -> None:
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register_extract_fn(extract_datetime)

    def register_extract_fn(self, fn: ExtractorFn) -> ExtractorFn:
        self.exctractor_fns.append(fn)
        return fn

    def extract_hints(self, callable: Callable) -> DocstringHints:
        docstring = inspect.getdoc(callable)
        if docstring is not None:
            docstring = parse_docstring(docstring)
            object_hint = docstring.short_description
            long_description = docstring.long_description
            param_to_hint = {param.arg_name: param.description for param in docstring.params if param.description}
        else:
            object_hint = None
            long_description = None
            param_to_hint = {}

        if isinstance(callable, type) and hasattr(callable, "__init__"):
            init_hints = self.extract_hints(callable.__init__)
            object_hint = object_hint or init_hints.object_hint
            long_description = long_description or init_hints.long_decsription
            param_to_hint.update(init_hints.param_to_hint)

        return DocstringHints(
            object_hint,
            long_description,
            param_to_hint,
        )

    def annotation_to_schema(self, annotation) -> SchemaNode:
        if annotation in [Any, None]:
            return UndefinedNode(original_annotation=annotation)

        annotation = normalize_annotation(annotation, concretize=True)

        if annotation in [int, float, str, bool]:
            return PrimitiveNode(primitive_type=annotation)  # type: ignore

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is None:
            # Object node, so we need to extract fields from the constructor signature.
            init_schema = self.extract_schema(annotation)
            return init_schema.call_schema

        if origin is Literal:
            return LiteralNode(options=list(args))

        if origin in [Union, UnionType]:
            # Reader may think that the second check is unnecessary, since Unions should have been
            # converted to `|` by the normalization step. Unfortunately, Literal[1] | str
            # will evaluate to Union, and not UnionType, so we have to check against both,
            # Union and UnionType.
            arg_schemas = []
            for arg in args:
                if arg is NoneType:
                    arg_schemas.append(None)
                else:
                    arg_schemas.append(self.annotation_to_schema(arg))
            return UnionNode(options=arg_schemas)

        # Normalization step has already converted all abstract classes to the corresponding concrete types.
        # So we can safely check against list and dict.
        if issubclass(origin, list):
            assert len(args) == 1, "Sequence type annotation should have exactly one argument."
            return SequenceNode(
                sequence_type=origin,
                item_schema=self.annotation_to_schema(args[0]),
            )
        if issubclass(origin, tuple):
            if len(args) == 2 and args[1] is Ellipsis:
                return SequenceNode(
                    sequence_type=origin,
                    item_schema=self.annotation_to_schema(args[0]),
                )
            return TupleNode(
                tuple_type=origin,
                item_schemas=[self.annotation_to_schema(arg) for arg in args],
            )
        if issubclass(origin, dict):
            assert len(args) == 2, "Mapping type annotation should have exactly two arguments."
            return MappingNode(
                mapping_type=origin,
                key_schema=self.annotation_to_schema(args[0]),
                value_schema=self.annotation_to_schema(args[1]),
            )

        raise ValueError(f"Unsupported type annotation: {annotation}")

    def extract_schema(self, callable: Callable) -> CallableSchema:
        r"""Extract schema from a callable."""
        try:
            return self._extract_schema_impl(callable)
        except Exception as e:
            raise SchemaExtractionError(f"Failed to extract schema for {callable}") from e

    def _extract_schema_impl(self, callable: Callable) -> CallableSchema:
        for fn in self.exctractor_fns:
            schema = fn(callable)
            if schema is not None:
                return schema

        try:
            signature = inspect.signature(callable, eval_str=True)
        except ValueError:
            # Some built-int types like `datetime.datetime` are not handled
            # correctly by `inspect.signature`. In such cases, we fall back to
            # `__init__` signature, but we still use the original docstring for hints.
            signature = inspect.signature(callable.__init__, eval_str=True)
            signature = strip_self(signature)

        return_type = signature.return_annotation

        hints = self.extract_hints(callable)

        if return_type is inspect.Signature.empty:
            return_schema = UndefinedNode(original_annotation=Any)
        else:
            return_schema = self.annotation_to_schema(return_type)

        object_fields = []
        for param in signature.parameters.values():
            if param.annotation is inspect.Signature.empty:
                arg_schema = UndefinedNode(original_annotation=Any)
            else:
                arg_schema = self.annotation_to_schema(param.annotation)
            field = ObjectField(
                param.name,
                arg_schema,
                hints.param_to_hint.get(param.name),
                param.default is param.empty,
            )
            object_fields.append(field)

        return CallableSchema(
            call_schema=ObjectNode(
                constructor_fn=callable,
                constructor_signature=signature,
                name=callable.__name__,
                fields=object_fields,
                hint=hints.object_hint,
            ),
            return_schema=return_schema,
            long_description=hints.long_decsription,
        )


DEFAULT_SCHEMA_EXTRACTOR = SchemaExtractor()
