import collections
import collections.abc
import typing
from dataclasses import dataclass
from datetime import datetime, timedelta

from attrs import frozen
from pydantic import BaseModel

from wanga.schema.extractor import default_schema_extractor
from wanga.schema.normalize import normalize_annotation, unpack_optional
from wanga.schema.schema import (
    CallableSchema,
    ObjectField,
    ObjectNode,
    PrimitiveNode,
    SequenceNode,
    UndefinedNode,
)


def test_normalize_schema():
    expected = {
        list: list,
        list[typing.Annotated[int, "tag"]]: list[int],
        list[typing.List[int]]: list[list[int]],
        list[typing.Tuple[int, str]]: list[tuple[int, str]],
        typing.Union[int, float]: int | float,
        typing.Optional[int]: int | None,
        typing.List: list,
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
    def foo(x: int, y: str = "hello"):  # noqa
        pass

    foo_schema = CallableSchema(
        return_schema=UndefinedNode(original_annotation=typing.Any),
        call_schema=ObjectNode(
            constructor_fn=foo,
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
            ],
        ),
    )

    assert default_schema_extractor.extract_schema(foo) == foo_schema

    def bar(x: typing.List[int]) -> int:  # noqa
        r"""Bar.

        Blah blah blah.

        Args:
            x: The x.
        """
        return 0

    bar_schema = CallableSchema(
        return_schema=PrimitiveNode(primitive_type=int),
        call_schema=ObjectNode(
            constructor_fn=bar,
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
            ],
        ),
    )

    assert default_schema_extractor.extract_schema(bar) == bar_schema

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
    )

    assert default_schema_extractor.extract_schema(Baz) == baz_schema

    @frozen
    class Qux:
        r"""I am Qux.

        I have attributes instead of arguments!

        Attributes:
            x: The x.
            baz: The baz.
        """

        x: int
        baz: Baz

    qux_schema = CallableSchema(
        return_schema=UndefinedNode(original_annotation=None),
        call_schema=ObjectNode(
            constructor_fn=Qux,
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
    )

    assert default_schema_extractor.extract_schema(Qux) == qux_schema

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
            name="Goo",
            hint="I am Goo.",
            fields=[
                ObjectField(
                    name="date",
                    schema=ObjectNode(
                        constructor_fn=datetime,
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
    )

    assert default_schema_extractor.extract_schema(Goo) == goo_schema

    class Hoo(BaseModel):
        r"""I am Hoo.

        I am a Pydantic model!
        And I use Numpy Doc format!.

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
            name="Hoo",
            hint="I am Hoo.",
            fields=[
                ObjectField(
                    name="delta",
                    schema=ObjectNode(
                        constructor_fn=timedelta,
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
    )

    assert default_schema_extractor.extract_schema(Hoo) == hoo_schema
