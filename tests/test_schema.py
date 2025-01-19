import collections
import collections.abc
import inspect
import platform
import typing
from dataclasses import dataclass
from datetime import datetime, timedelta
from textwrap import dedent

from attrs import frozen
from pydantic import BaseModel

from wanga.schema.extractor import DEFAULT_SCHEMA_EXTRACTOR
from wanga.schema.jsonschema import JsonSchemaFlavor
from wanga.schema.normalize import normalise_aliases, normalize_annotation, unpack_optional  # noqa: F401
from wanga.schema.schema import (
    CallableSchema,
    LiteralNode,
    ObjectField,
    ObjectNode,
    PrimitiveNode,
    SequenceNode,
    UndefinedNode,
    UnionNode,
)
from wanga.schema.utils import strip_self


def test_normalize_schema():
    expected = {
        list: list,
        list[typing.Annotated[int, "tag"]]: list[int],
        list[typing.List[int]]: list[list[int]],
        list[typing.Tuple[int, str]]: list[tuple[int, str]],
        typing.Union[int, float]: int | float,
        typing.Optional[int]: int | None,
        typing.List: list,
        typing.Union[typing.Union[int, float], str]: int | float | str,
        (typing.Literal[1] | typing.Literal[2] | typing.Literal[3]): typing.Literal[1, 2, 3],
        (typing.Literal[1, 2] | typing.Union[typing.Literal[2, 3], typing.Literal[3, 4]]): (
            typing.Literal[1, 2, 3, 4]
        ),
    }
    for annotation, result in expected.items():
        assert normalize_annotation(annotation) == result


def test_unpack_optional():
    expected = {
        typing.Optional[int]: int,
        int | float | None: int | float,
        int: None,
    }
    for annotation, result in expected.items():
        assert unpack_optional(annotation) == result


def test_concretize_schema():
    expected = {
        typing.List[collections.abc.Iterable[int]]: list[list[int]],
        typing.List[int]: list[int],
        int: int,
        collections.abc.Mapping[str, typing.List[int]]: dict[str, list[int]],
    }
    for annotation, result in expected.items():
        assert normalize_annotation(annotation, concretize=True) == result


def test_extract_schema():
    def foo(x: int, y: str = "hello", z: tuple[int, ...] = ()):  # noqa
        pass

    foo_schema = CallableSchema(
        return_schema=UndefinedNode(original_annotation=typing.Any),
        call_schema=ObjectNode(
            constructor_fn=foo,
            constructor_signature=inspect.signature(foo),
            name="foo",
            hint=None,
            fields=[
                ObjectField(
                    name="x",
                    schema=PrimitiveNode(primitive_type=int),
                    required=True,
                    hint=None,
                ),
                ObjectField(
                    name="y",
                    schema=PrimitiveNode(primitive_type=str),
                    required=False,
                    hint=None,
                ),
                ObjectField(
                    name="z",
                    schema=SequenceNode(
                        sequence_type=tuple,
                        item_schema=PrimitiveNode(primitive_type=int),
                    ),
                    required=False,
                    hint=None,
                ),
            ],
        ),
        long_description=None,
    )

    assert DEFAULT_SCHEMA_EXTRACTOR.extract_schema(foo) == foo_schema

    def bar(x: typing.List[int], y: typing.Literal["hehe"] | float) -> int:  # noqa
        r"""Bar.

        Blah blah blah.

        Args:
            x: The x.
            y: Hard example.
        """
        return 0

    bar_schema = CallableSchema(
        return_schema=PrimitiveNode(primitive_type=int),
        call_schema=ObjectNode(
            constructor_fn=bar,
            constructor_signature=inspect.signature(bar),
            name="bar",
            hint="Bar.",
            fields=[
                ObjectField(
                    name="x",
                    schema=SequenceNode(
                        sequence_type=list,
                        item_schema=PrimitiveNode(primitive_type=int),
                    ),
                    required=True,
                    hint="The x.",
                ),
                ObjectField(
                    name="y",
                    schema=UnionNode(
                        [
                            PrimitiveNode(primitive_type=float),
                            LiteralNode(options=["hehe"]),
                        ]
                    ),
                    required=True,
                    hint="Hard example.",
                ),
            ],
        ),
        long_description="Blah blah blah.",
    )

    assert DEFAULT_SCHEMA_EXTRACTOR.extract_schema(bar) == bar_schema

    class Baz:
        r"""I am Baz."""

        def __init__(self, x: int, y):
            r"""Init Baz.

            Args:
                x: The x.
                y: The y.
            """

    baz_schema = CallableSchema(
        return_schema=UndefinedNode(original_annotation=typing.Any),
        call_schema=ObjectNode(
            constructor_fn=Baz,
            constructor_signature=inspect.signature(Baz),
            name="Baz",
            hint="I am Baz.",
            fields=[
                ObjectField(
                    name="x",
                    schema=PrimitiveNode(primitive_type=int),
                    required=True,
                    hint="The x.",
                ),
                ObjectField(
                    name="y",
                    schema=UndefinedNode(original_annotation=typing.Any),
                    required=True,
                    hint="The y.",
                ),
            ],
        ),
        long_description=None,
    )

    assert DEFAULT_SCHEMA_EXTRACTOR.extract_schema(Baz) == baz_schema

    @frozen
    class Qux:
        r"""I am Qux.

        I have attributes instead of arguments!

        Attributes:
            x: The x.
            baz: The baz.
        """

        x: "int"
        baz: Baz

    qux_schema = CallableSchema(
        return_schema=UndefinedNode(original_annotation=None),
        call_schema=ObjectNode(
            constructor_fn=Qux,
            constructor_signature=inspect.signature(Qux, eval_str=True),
            name="Qux",
            hint="I am Qux.",
            fields=[
                ObjectField(
                    name="x",
                    schema=PrimitiveNode(primitive_type=int),
                    required=True,
                    hint="The x.",
                ),
                ObjectField(
                    name="baz",
                    schema=baz_schema.call_schema,
                    required=True,
                    hint="The baz.",
                ),
            ],
        ),
        long_description="I have attributes instead of arguments!",
    )

    assert DEFAULT_SCHEMA_EXTRACTOR.extract_schema(Qux) == qux_schema

    @dataclass
    class Goo:
        r"""I am Goo.

        I am a dataclass, and I use the stupid ReST docstring syntax!

        :param date: Ohoho, good luck.
        """

        date: datetime

    goo_schema = CallableSchema(
        return_schema=UndefinedNode(original_annotation=None),
        call_schema=ObjectNode(
            constructor_fn=Goo,
            constructor_signature=inspect.signature(Goo),
            name="Goo",
            hint="I am Goo.",
            fields=[
                ObjectField(
                    name="date",
                    schema=ObjectNode(
                        constructor_fn=datetime,
                        constructor_signature=strip_self(inspect.signature(datetime.__init__)),
                        name="datetime",
                        hint=None,
                        fields=[
                            ObjectField(
                                name="year",
                                schema=PrimitiveNode(primitive_type=int),
                                required=True,
                                hint=None,
                            ),
                            ObjectField(
                                name="month",
                                schema=PrimitiveNode(primitive_type=int),
                                required=True,
                                hint=None,
                            ),
                            ObjectField(
                                name="day",
                                schema=PrimitiveNode(primitive_type=int),
                                required=True,
                                hint=None,
                            ),
                            ObjectField(
                                name="hour",
                                schema=PrimitiveNode(primitive_type=int),
                                required=False,
                                hint=None,
                            ),
                            ObjectField(
                                name="minute",
                                schema=PrimitiveNode(primitive_type=int),
                                required=False,
                                hint=None,
                            ),
                            ObjectField(
                                name="second",
                                schema=PrimitiveNode(primitive_type=int),
                                required=False,
                                hint=None,
                            ),
                        ],
                    ),
                    required=True,
                    hint="Ohoho, good luck.",
                ),
            ],
        ),
        long_description="I am a dataclass, and I use the stupid ReST docstring syntax!",
    )

    assert DEFAULT_SCHEMA_EXTRACTOR.extract_schema(Goo) == goo_schema

    class Hoo(BaseModel):
        r"""I am Hoo.

        I am a Pydantic model!
        And I use Numpy Doc format!

        Parameters
        ----------
        delta : timedelta
            Thou shall not pass!
        """

        delta: timedelta

    hoo_schema = CallableSchema(
        return_schema=UndefinedNode(original_annotation=None),
        call_schema=ObjectNode(
            constructor_fn=Hoo,
            constructor_signature=inspect.signature(Hoo),
            name="Hoo",
            hint="I am Hoo.",
            fields=[
                ObjectField(
                    name="delta",
                    schema=ObjectNode(
                        constructor_fn=timedelta,
                        constructor_signature=strip_self(inspect.signature(timedelta.__init__)),
                        name="timedelta",
                        hint=None,
                        fields=[
                            ObjectField(
                                name="days",
                                schema=PrimitiveNode(primitive_type=int),
                                required=False,
                                hint=None,
                            ),
                            ObjectField(
                                name="seconds",
                                schema=PrimitiveNode(primitive_type=int),
                                required=False,
                                hint=None,
                            ),
                        ],
                    ),
                    required=True,
                    hint="Thou shall not pass!",
                ),
            ],
        ),
        long_description="I am a Pydantic model!\nAnd I use Numpy Doc format!",
    )

    assert DEFAULT_SCHEMA_EXTRACTOR.extract_schema(Hoo) == hoo_schema


def test_type_statement():
    _, minor, _ = platform.python_version_tuple()
    if int(minor) < 12:
        return

    expr = r"""
    type A = int | float

    assert normalise_aliases(A) == int | float  # type: ignore

    def foo() -> A:  # type: ignore
        pass

    expected = UnionNode([PrimitiveNode(primitive_type=int), PrimitiveNode(primitive_type=float)])
    assert DEFAULT_SCHEMA_EXTRACTOR.extract_schema(foo).return_schema == expected

    def bar() -> list[A]:  # type: ignore
        pass

    expected = SequenceNode(
        sequence_type=list,
        item_schema=UnionNode([PrimitiveNode(primitive_type=int), PrimitiveNode(primitive_type=float)]),
    )
    assert DEFAULT_SCHEMA_EXTRACTOR.extract_schema(bar).return_schema == expected
    """

    expr = dedent(expr)
    exec(expr)


def test_json_schema():
    @frozen
    class Inner:
        """Inner.

        Long description of Inner.

        Attributes:
            x: The x.
        """

        x: int

    def foo(
        a: int,
        b: str,
        c: Inner,
        d: tuple[int, ...] = (),
        e: typing.Literal["x", "y"] = "x",
        f: str | int = 1,
    ):
        r"""Foo!

        Long description of foo.

        Args:
            a: The a.
            b: The b.
            c: The c.
        """

    expected_json_schema = {
        "name": "foo",
        "description": "Foo!\n\nLong description of foo.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "The a."},
                "b": {"type": "string", "description": "The b."},
                "c": {
                    "type": "object",
                    "properties": {"x": {"type": "integer", "description": "The x."}},
                    "required": ["x"],
                    "description": "The c.\n\nInner.",
                },
                "d": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "e": {
                    "type": "string",
                    "enum": ["x", "y"],
                },
                "f": {
                    "type": ["string", "integer"],
                },
            },
            "required": ["a", "b", "c"],
        },
    }

    core_schema = DEFAULT_SCHEMA_EXTRACTOR.extract_schema(foo)
    json_schema = core_schema.json_schema(JsonSchemaFlavor.OPENAI, include_long_description=True)
    assert json_schema == expected_json_schema


def test_eval():
    @frozen
    class Huhu:
        meee: int

    @frozen
    class Hehe:
        hehehe: int
        hohoho: str
        huhuhu: Huhu

    def foo(
        x: float,
        /,
        y: int = 3,
        *,
        z: typing.Literal["a", "b"],
        hehe: Hehe | None = None,
    ):
        assert hehe is not None
        return hehe.huhuhu.meee

    json_input = {
        "x": 1,
        "y": 2,
        "z": "a",
        "hehe": {"hehehe": 3, "hohoho": "haha", "huhuhu": {"meee": 4}},
    }

    core_schema = DEFAULT_SCHEMA_EXTRACTOR.extract_schema(foo)
    assert core_schema.eval(json_input) == 4
