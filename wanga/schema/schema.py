from collections.abc import Mapping, Sequence
from inspect import Signature
from types import NoneType
from typing import Callable, NoReturn

from attrs import evolve, frozen

from ..common import JSON
from .jsonschema import (
    AnthropicCallableSchema,
    ArrayJsonSchema,
    CallableJsonSchema,
    EnumJsonSchema,
    JsonSchema,
    JsonSchemaFlavor,
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
    "SchemaValidationError",
    "SequenceNode",
    "TupleNode",
    "UndefinedNode",
    "UnionNode",
    "UnsupportedSchemaError",
]

_type_to_jsonname: dict[type | None, LeafTypeName] = {
    None: "null",
    bool: "boolean",
    float: "number",
    int: "integer",
    str: "string",
}


class UnsupportedSchemaError(ValueError):
    pass


class SchemaValidationError(ValueError):
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

    def eval(self, value: JSON):
        r"""Evaluate the value against the schema."""
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
        raise UnsupportedSchemaError("JSON schema cannot be generated for missing or undefined annotations.")

    def eval(self, value: JSON) -> NoReturn:
        raise UnsupportedSchemaError("Cannot evaluate undefined schema.")


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
            type=_type_to_jsonname[self.primitive_type],
        )
        if parent_hint:
            result["description"] = parent_hint
        return result

    def eval(self, value: JSON) -> int | float | str | bool:
        if not isinstance(value, self.primitive_type):
            if self.primitive_type is float and isinstance(value, int):
                return float(value)
            else:
                raise SchemaValidationError(f"Expected {self.primitive_type}, got {value}")
        return value


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

    def eval(self, value: JSON) -> list[JSON]:
        if not isinstance(value, list):
            raise SchemaValidationError(f"Expected list, got {value}")
        return [self.item_schema.eval(item) for item in value]


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
        raise UnsupportedSchemaError("JSON schema cannot be generated for heterogeneous tuple types.")

    def eval(self, value: JSON) -> tuple:
        if not isinstance(value, list):
            raise SchemaValidationError(f"Expected list, got {value}")
        if len(value) != len(self.item_schemas):
            raise SchemaValidationError(f"Expected tuple of length {len(self.item_schemas)}, got {len(value)}")
        return tuple(item_schema.eval(item) for item_schema, item in zip(self.item_schemas, value))


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
        raise UnsupportedSchemaError("JSON schema cannot be generated for Mapping types.")

    def eval(self, value: JSON) -> NoReturn:
        raise UnsupportedSchemaError("Cannot evaluate Mapping schema.")


@frozen
class UnionNode(SchemaNode):
    r"""Node corresponding to `Union` and `Optional` types.

    Attributes:
        options: The schemas of the options in the union.
            May be None in case of optional types.
    """

    options: list[SchemaNode | None]

    @property
    def is_optional(self) -> bool:
        return None in self.options

    @property
    def is_primitive(self) -> bool:
        if len(self.options) == 1:
            return True
        if len(self.options) == 2 and None in self.options:
            return True
        return all(option is None or isinstance(option, PrimitiveNode) for option in self.options)

    def json_schema(self, parent_hint: str | None = None) -> JsonSchema:
        if not self.is_primitive:
            raise UnsupportedSchemaError("JSON schema cannot be generated for non-trivial Union types.")
        if self.is_optional:
            options: list[SchemaNode | None] = [option for option in self.options if option is not None]
            if len(options) == 1:
                assert options[0] is not None
                return options[0].json_schema(parent_hint)
            return UnionNode(options).json_schema(parent_hint)
        type_names = [
            _type_to_jsonname[option.primitive_type]  # type: ignore
            for option in self.options
            if option is not None
        ]
        if "number" in type_names and "integer" in type_names:
            type_names.remove("integer")
        if len(type_names) == 1:
            type_names = type_names[0]
        result = LeafJsonSchema(
            type=type_names,  # type: ignore
        )
        if parent_hint:
            result["description"] = parent_hint
        return result

    def eval(self, value: JSON) -> JSON:
        if not self.is_primitive:
            raise UnsupportedSchemaError("Cannot evaluate non-primitive Union schema.")
        for option in self.options:
            if option is None:
                if value is None:
                    return None
                else:
                    continue
            try:
                return option.eval(value)
            except SchemaValidationError:
                continue
        raise SchemaValidationError(f"Value {value} does not match any of the options: {self.options}")


@frozen
class LiteralNode(SchemaNode):
    r"""Node corresponding to the `Literal` type.

    Attributes:
        options: The value of the literal.
    """

    options: list[int | float | str | bool]

    def json_schema(self, parent_hint: str | None = None) -> EnumJsonSchema:
        if not all(isinstance(option, str) for option in self.options):
            raise UnsupportedSchemaError("JSON schema can only be generated for string literal types.")
        result = EnumJsonSchema(type="string", enum=self.options)  # type: ignore
        if parent_hint:
            result["description"] = parent_hint
        return result

    def eval(self, value: JSON) -> int | float | str | bool:
        if value not in self.options:
            raise SchemaValidationError(f"Value {value} does not match any of the options: {self.options}")
        return value


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
        constructor_fn: Callable that can be used to construct an object.
        constructor_signature: Used to properly dispatch positional and
            keyword-only args during evaluation.
        name: Name of the object.
        fields: The fields of the object.
        hint: Hint extracted from the docstring.
    """

    constructor_fn: Callable
    constructor_signature: Signature
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

    def eval(self, value: JSON):
        if not isinstance(value, dict):
            raise SchemaValidationError(f"Expected object, got {value}")
        args = []
        kwargs = {}
        pos_only_args = {
            param.name
            for param in self.constructor_signature.parameters.values()
            if param.kind == param.POSITIONAL_ONLY
        }
        missing_args = {field.name for field in self.fields if field.required}
        name_to_schema = {field.name: field.schema for field in self.fields}
        for arg_name, arg_value in value.items():
            arg_schema = name_to_schema.get(arg_name)
            if arg_schema is None:
                raise SchemaValidationError(f"Unexpected field: {arg_name}")
            evaled_arg_value = arg_schema.eval(arg_value)
            if arg_name in pos_only_args:
                args.append(evaled_arg_value)
            else:
                kwargs[arg_name] = evaled_arg_value
            if arg_name in missing_args:
                missing_args.remove(arg_name)
        if missing_args:
            raise SchemaValidationError(f"Missing required fields: {missing_args}")
        return self.constructor_fn(*args, **kwargs)


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

    @property
    def name(self) -> str:
        return self.call_schema.name

    def json_schema(self, flavor: JsonSchemaFlavor, include_long_description: bool = False) -> CallableJsonSchema:
        r"""Extract JSON Schema to use with the LLM function call APIs.

        Args:
            flavor: Flavor of the JSON Schema syntax accepted by the target API.
            include_long_description: Whether to include `long_description` to the
                `description` field of the resulting JSON Schema.
                Should be set to True for tools, and to False for object constructions.
        """
        full_description = []
        if self.call_schema.hint:
            full_description.append(self.call_schema.hint)
        if self.long_description and include_long_description:
            full_description.append(self.long_description)
        full_description = "\n\n".join(full_description)
        if flavor is JsonSchemaFlavor.ANTHROPIC:
            result = AnthropicCallableSchema(
                name=self.name,
                input_schema=evolve(self.call_schema, hint=None).json_schema(),
            )
            if full_description:
                result["description"] = full_description
        elif flavor is JsonSchemaFlavor.OPENAI:
            result = OpenAICallableSchema(
                name=self.name,
                parameters=evolve(self.call_schema, hint=None).json_schema(),
            )
            if full_description:
                result["description"] = full_description
        else:
            raise ValueError(f"Unknown JSON schema flavour: {flavor}")
        return result

    def eval(self, value: JSON):
        r"""Call the callable with the given JSON value."""
        return self.call_schema.eval(value)
