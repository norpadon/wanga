import collections
import collections.abc
import typing

from attrs import frozen

from wanga.schema.extract import *
from wanga.schema.normalize import *
from wanga.schema.schema import *


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
    def foo(x: int, y: str = "hello"):
        pass

    foo_schema = CallableSchema(
        return_schema=UndefinedNode(original_annotation=typing.Any),
        call_schema=ObjectNode(
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

    assert extract_schema(foo) == foo_schema

    def bar(x: typing.List[int]) -> int:
        r"""Bar.

        Blah blah blah.

        Args:
            x: The x.
        """
        return 0

    bar_schema = CallableSchema(
        return_schema=PrimitiveNode(primitive_type=int),
        call_schema=ObjectNode(
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

    assert extract_schema(bar) == bar_schema

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

    assert extract_schema(Baz) == baz_schema

    @frozen
    class Qux:
        """I am Qux.

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

    assert extract_schema(Qux) == qux_schema
