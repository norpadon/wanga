import collections
import collections.abc
import platform
import typing  # noqa: F401
from types import NoneType, UnionType
from typing import Annotated, Literal, Union, get_args, get_origin

if int(platform.python_version_tuple()[1]) >= 12:
    from typing import TypeAliasType  # type: ignore
else:

    class TypeAliasType:
        pass


from .utils import TypeAnnotation

__all__ = [
    "normalize_annotation",
    "unpack_optional",
]


def _fold_or(annotations: collections.abc.Sequence[TypeAnnotation]) -> type[UnionType]:
    result = annotations[0]
    for annotation in annotations[1:]:
        result = result | annotation
    return result


def normalise_aliases(annotation: TypeAnnotation) -> TypeAnnotation:
    if isinstance(annotation, TypeAliasType):
        return annotation.__value__  # type: ignore
    return annotation


def unpack_optional(annotation: TypeAnnotation) -> type[UnionType] | None:
    r"""Unpack Optional[T] to its inner type T.

    Returns None if the annotation is not Optional[T].

    Examples:
    >>> unpack_optional(typing.Optional[int])
    <class 'int'>
    >>> unpack_optional(int | float | None)
    int | float
    >>> unpack_optional(int) is None
    True
    """
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin not in [Union, UnionType]:
        return None
    if NoneType not in args:
        return None
    result = tuple(arg for arg in args if arg is not NoneType)
    return _fold_or(result)


def normalize_literals(annotation: TypeAnnotation) -> TypeAnnotation:
    r"""Merges literals within unions.

    Examples:
    >>> normalize_literals(typing.Literal[1] | typing.Literal[2])
    typing.Literal[1, 2]
    >>> normalize_literals(typing.Literal[1] | typing.Literal[2] | str)
    typing.Union[str, typing.Literal[1, 2]]
    """
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in [Literal, None]:
        return annotation
    args = tuple(normalize_literals(arg) for arg in args)
    if origin in [Union, UnionType]:
        literals = []
        non_literals = []
        for arg in args:
            if get_origin(arg) is Literal:
                literals.extend(get_args(arg))
            else:
                non_literals.append(arg)
        new_args = list(non_literals)
        if literals:
            new_args.append(Literal[tuple(literals)])  # type: ignore
        return _fold_or(new_args)
    return origin[args]


ABSTRACT_TO_CONCRETE = {
    collections.abc.Set: set,
    collections.abc.ByteString: bytes,
    collections.abc.Mapping: dict,
    collections.abc.MutableMapping: dict,
    collections.abc.MutableSequence: list,
    collections.abc.MutableSet: set,
    collections.abc.Sequence: list,
    collections.abc.MappingView: dict,
    collections.abc.ValuesView: list,
    collections.abc.KeysView: list,
    collections.abc.Iterable: list,
    collections.abc.Iterator: list,
    collections.abc.Container: list,
    collections.abc.Collection: list,
    collections.abc.Sized: list,
    collections.abc.Reversible: list,
}


def _normalize_annotation_rec(annotation: TypeAnnotation, concretize: bool = False) -> TypeAnnotation:
    annotation = normalise_aliases(annotation)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        return annotation
    if args:
        args = tuple(_normalize_annotation_rec(arg, concretize=concretize) for arg in args)
    if origin is Annotated:
        return args[0]
    if origin in [Union, UnionType]:
        return _fold_or(args)
    if concretize:
        origin = ABSTRACT_TO_CONCRETE.get(origin, origin)
    if args:
        return origin[args]
    return origin


def normalize_annotation(annotation: TypeAnnotation, concretize: bool = False) -> TypeAnnotation:
    r"""Normalize a type annotation to a standard form.

    Strips `Annotated` tags and replaces generic aliases with corresponding generic types.
    Replaces `Optional` and `Union` with `|`.

    if `concretize` is True, replaces abstract types with concrete types.

    Examples:
    >>> normalize_annotation(typing.List[int])
    list[int]
    >>> normalize_annotation(typing.Annotated[str, 'tag'])
    <class 'str'>
    >>> normalize_annotation(typing.Union[int, str])
    int | str
    >>> normalize_annotation(collections.abc.Sequence[int])
    collections.abc.Sequence[int]
    >>> normalize_annotation(collections.abc.Sequence[int], concretize=True)
    list[int]
    >>> normalize_annotation(typing.Literal[1] | typing.Literal[2])
    typing.Literal[1, 2]
    """
    result = normalize_literals(annotation)
    result = _normalize_annotation_rec(result, concretize=concretize)
    return result
