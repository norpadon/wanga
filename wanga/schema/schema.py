from collections.abc import Mapping, Sequence
from types import NoneType
from typing import Callable, Literal, TypeAlias

from attrs import evolve, frozen

from .jsonschema import (
    AnthropicCallableSchema,
    ArrayJsonSchema,
    CallableJsonSchema,
    EnumJsonSchema,
    JsonSchema,
    JsonSchemaFlavour,
    LeafJsonSchema,
    LeafTypeName,
    ObjectJsonSchema,
    OpenAICallableSchema,
)
from .utils import TypeAnnotation

__all__ = [
    "CallableSchema",
    "JSON",
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


JSON: TypeAlias = int | float | str | None | dict[str, "JSON"] | list["JSON"]


type_to_jsonname: dict[type | None, LeafTypeName] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    None: "null",
}


class JsonSchemaGenerationError(ValueError):
    pass


@frozen
class SchemaNode:
    r"""Base class for schema nodes."""

    def json_schema(self, parent_hint: str | None = None) -> JsonSchema:
        r"""Returns the JSON schema of the node to use in the LLM function call APIs.

        Args:
            parent_hint: Hint from the parent object.
        """
        raise NotImplementedError


@frozen
class UndefinedNode(SchemaNode):
    r"""Node corresponding to missing, `Any`, or `None` annotations.

    Atributes
        original_annotation: `Any` if the annotation is missing or `Any`,
            `None` if the annotation is `None`.
    """

    original_annotation: NoneType | TypeAnnotation

    def json_schema(self, parent_hint: str | None = None) -> LeafJsonSchema:
        raise JsonSchemaGenerationError(
            "JSON schema cannot be generated for missing or undefined annotations."
        )


@frozen
class PrimitiveNode(SchemaNode):
    r"""Node corresponding to primitive types.

    Primitime types are `int`, `float`, `str`, and `bool`.

    Attributes:
        primitive_type: The primitive type.
    """

    primitive_type: type[int] | type[float] | type[str] | type[bool]

    def json_schema(self, parent_hint: str | None = None) -> LeafJsonSchema:
        result = LeafJsonSchema(
            type=type_to_jsonname[self.primitive_type],
        )
        if parent_hint:
            result["description"] = parent_hint
        return result


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

    def json_schema(self, parent_hint: str | None = None) -> ArrayJsonSchema:
        result = ArrayJsonSchema(type="array", items=self.item_schema.json_schema())
        if parent_hint:
            result["description"] = parent_hint
        return result


@frozen
class TupleNode(SchemaNode):
    r"""Node corresponding to tuples and named tuples.

    Attributes:
        tuple_type: The tuple type (`tuple`, or a subclass of `typing.NamedTuple`).
        item_schemas: The schemas of the items in the tuple.
    """

    tuple_type: type[tuple]
    item_schemas: list[SchemaNode]

    def json_schema(self, parent_hint: str | None = None) -> JsonSchema:
        raise JsonSchemaGenerationError(
            "JSON schema cannot be generated for heterogeneous tuple types."
        )


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

    def json_schema(self, parent_hint: str | None = None) -> JsonSchema:
        raise JsonSchemaGenerationError(
            "JSON schema cannot be generated for Mapping types."
        )


@frozen
class UnionNode(SchemaNode):
    r"""Node corresponding to `Union` and `Optional` types.

    Attributes:
        options: The schemas of the options in the union.
            May be None in case of optional types.
    """

    options: list[SchemaNode | None]

    def json_schema(self, parent_hint: str | None = None) -> JsonSchema:
        if all(
            option is None or isinstance(option, PrimitiveNode)
            for option in self.options
        ):
            type_names = {
                type_to_jsonname[option.primitive_type]  # type: ignore
                for option in self.options
                if option is not None
            }
            if "number" in type_names and "integer" in type_names:
                type_names.remove("integer")
            type_names = list(type_names)
            if len(type_names) == 1:
                type_names = type_names[0]
            result = LeafJsonSchema(
                type=type_names,  # type: ignore
            )
            if parent_hint:
                result["description"] = parent_hint
            return result
        raise JsonSchemaGenerationError(
            "JSON schema cannot be generated for non-trivial Union types."
        )


@frozen
class LiteralNode(SchemaNode):
    r"""Node corresponding to the `Literal` type.

    Attributes:
        options: The value of the literal.
    """

    options: list[int | float | str | bool]

    def json_schema(self, parent_hint: str | None = None) -> EnumJsonSchema:
        if not all(isinstance(option, str) for option in self.options):
            raise JsonSchemaGenerationError(
                "JSON schema can only be generated for string literal types."
            )
        result = EnumJsonSchema(type="string", enum=self.options)  # type: ignore
        if parent_hint:
            result["description"] = parent_hint
        return result


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

    def json_schema(self) -> JsonSchema:
        return self.schema.json_schema(parent_hint=self.hint)


@frozen
class ObjectNode(SchemaNode):
    r"""Node corresponding to composite types.

    Represents the signature of the constructor.

    Attributes:
        name: Name of the object.
        fields: The fields of the object.
        hint: Hint extracted from the docstring.
    """

    constructor_fn: Callable
    name: str
    fields: list[ObjectField]
    hint: str | None

    def json_schema(self, parent_hint: str | None = None) -> ObjectJsonSchema:
        result = ObjectJsonSchema(
            type="object",
            properties={field.name: field.json_schema() for field in self.fields},
            required=[field.name for field in self.fields if field.required],
        )
        joint_hint = []
        if parent_hint:
            joint_hint.append(parent_hint)
        if self.hint:
            joint_hint.append(self.hint)
        joint_hint = "\n\n".join(joint_hint)
        if joint_hint:
            result["description"] = joint_hint
        return result


@frozen
class CallableSchema:
    r"""Complete schema of a function or a class.

    Attributes:
        call_schema: Schema of the function call.
        return_schema: Schema of the return value.
            None if the function returns None.
        long_description: Long description extracted from the docstring.
            It is used to pass tool descriptions to LLMs. It is not used
            for return values.
    """

    call_schema: ObjectNode
    return_schema: SchemaNode
    long_description: str | None

    def json_schema(
        self, flavour: JsonSchemaFlavour, include_long_description: bool = False
    ) -> CallableJsonSchema:
        full_description = []
        if self.call_schema.hint:
            full_description.append(self.call_schema.hint)
        if self.long_description and include_long_description:
            full_description.append(self.long_description)
        full_description = "\n\n".join(full_description)
        if flavour is JsonSchemaFlavour.ANTHROPIC:
            result = AnthropicCallableSchema(
                name=self.call_schema.name,
                input_schema=evolve(self.call_schema, hint=None).json_schema(),
            )
            if full_description:
                result["description"] = full_description
        elif flavour is JsonSchemaFlavour.OPENAI:
            result = OpenAICallableSchema(
                name=self.call_schema.name,
                parameters=evolve(self.call_schema, hint=None).json_schema(),
            )
            if full_description:
                result["description"] = full_description
        else:
            raise ValueError(f"Unknown JSON schema flavour: {flavour}")
        return result
