from __future__ import annotations

from enum import Enum
from typing import Literal, TypeAlias, TypedDict

__all__ = [
    "JsonSchemaFlavor",
    "LeafJsonSchema",
    "EnumJsonSchema",
    "ObjectJsonSchema",
    "AnthropicCallableSchema",
    "OpenAICallableSchema",
    "CallableJsonSchema",
    "JsonSchema",
    "ArrayJsonSchema",
    "LeafTypeName",
]


LeafTypeName: TypeAlias = Literal["string", "number", "integer", "boolean", "null"]


class LeafJsonSchema(TypedDict, total=False):
    type: list[LeafTypeName] | LeafTypeName
    description: str


class EnumJsonSchema(TypedDict, total=False):
    type: Literal["string"]
    enum: list[str]
    description: str


class ObjectJsonSchema(TypedDict, total=False):
    type: Literal["object"]
    properties: dict[str, JsonSchema]
    required: list[str]
    description: str


class ArrayJsonSchema(TypedDict, total=False):
    type: Literal["array"]
    items: JsonSchema
    description: str


JsonSchema: TypeAlias = LeafJsonSchema | EnumJsonSchema | ObjectJsonSchema | ArrayJsonSchema


class AnthropicCallableSchema(TypedDict, total=False):
    name: str
    description: str
    input_schema: ObjectJsonSchema


class OpenAICallableSchema(TypedDict, total=False):
    name: str
    description: str
    parameters: ObjectJsonSchema


class JsonSchemaFlavor(Enum):
    r"""Top-level layout of the JSON schema as accepted by different LLMS."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


CallableJsonSchema: TypeAlias = AnthropicCallableSchema | OpenAICallableSchema
