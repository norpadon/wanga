from collections.abc import Mapping, Sequence
from types import NoneType
from typing import Any

from attrs import frozen

__all__ = [
    "CallableSchema",
    "MappingNode",
    "ObjectField",
    "ObjectNode",
    "PrimitiveNode",
    "SchemaNode",
    "SequenceNode",
    "TupleNode",
    "UndefinedNode",
    "UnionNode",
]


@frozen
class SchemaNode:
    r"""Base class for schema nodes."""

    pass


@frozen
class UndefinedNode(SchemaNode):
    r"""Node corresponding to missing, `Any`, or `None` annotations.

    Atributes
        original_annotation: `Any` if the annotation is missing or `Any`,
            `None` if the annotation is `None`.
    """

    original_annotation: NoneType | Any


@frozen
class PrimitiveNode(SchemaNode):
    r"""Node corresponding to primitive types.

    Primitime types are `int`, `float`, `str`, and `bool`.

    Attributes:
        primitive_type: The primitive type.
    """

    primitive_type: type[int] | type[float] | type[str] | type[bool]


@frozen
class SequenceNode(SchemaNode):
    r"""Node corresponding to generic homogeneous sequence types.
        (Does not include tuples and strings)

    Attributes:
        sequence_type: The sequence type.
        item_schema: The schema of the items in the sequence.
    """

    sequence_type: type[Sequence]
    item_schema: SchemaNode


@frozen
class TupleNode(SchemaNode):
    r"""Node corresponding to tuples and named tuples.

    Attributes:
        tuple_type: The tuple type (`tuple`, or a subclass of `typing.NamedTuple`).
        item_schemas: The schemas of the items in the tuple.
    """

    tuple_type: type[tuple]
    item_schemas: list[SchemaNode]


@frozen
class MappingNode(SchemaNode):
    r"""Node corresponding to generic mapping types.

    Attributes:
        mapping_type: The mapping type (e.g. `dict` or `collections.defaultdict`).
        key_schema: The schema of the keys in the mapping.
        value_schema: The schema of the values in the mapping.
    """

    mapping_type: type[Mapping]
    key_schema: SchemaNode
    value_schema: SchemaNode


@frozen
class UnionNode(SchemaNode):
    r"""Node corresponding to `Union` and `Optional` types.

    Attributes:
        options: The schemas of the options in the union.
            May be None in case of optional types.
    """

    options: list[SchemaNode | None]


@frozen
class ObjectField:
    r"""Field in an object schema.

    Arguments:
        name: Name of the field.
        schema: Schema of the field.
        hint: Hint extracted from the docstring.
        required: Whether the field is optional or required.
    """

    name: str
    schema: SchemaNode
    hint: str | None
    required: bool


@frozen
class ObjectNode(SchemaNode):
    r"""Node corresponding to composite types.

    Represents the signature of the constructor.

    Attributes:
        name: Name of the object.
        fields: The fields of the object.
        hint: Hint extracted from the docstring.
    """

    name: str
    fields: list[ObjectField]
    hint: str | None


@frozen
class CallableSchema:
    r"""Complete schema of a function of a class.

    Attributes:
        call_schema: Schema of the function call.
        return_schema: Schema of the return value.
            None if the function returns None.
    """

    call_schema: ObjectNode
    return_schema: SchemaNode
