import collections
import collections.abc
import typing  # noqa
from types import NoneType, UnionType
from typing import Annotated, Union, get_args, get_origin

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


# Those aliases are automatically resolved by the `typing.get_origin`, so there is no
# direct need to handle them explicitly, by we still include them here for the reference
# purposes.
#
# GENERIC_ALIASES = {
#     # Basic aliases
#     typing.Dict: dict,
#     typing.List: list,
#     typing.Set: set,
#     typing.FrozenSet: frozenset,
#     typing.Tuple: tuple,
#     typing.Type: type,
#     # collections aliases
#     typing.DefaultDict: collections.defaultdict,
#     typing.OrderedDict: collections.OrderedDict,
#     typing.ChainMap: collections.ChainMap,
#     typing.Counter: collections.Counter,
#     typing.Deque: collections.deque,
#     # re aliases
#     typing.Pattern: re.Pattern,
#     typing.Match: re.Match,
#     typing.Text: str,
#     # collections.abc aliases
#     typing.AbstractSet: collections.abc.Set,
#     typing.ByteString: collections.abc.ByteString,
#     typing.Collection: collections.abc.Collection,
#     typing.Container: collections.abc.Container,
#     typing.ItemsView: collections.abc.ItemsView,
#     typing.KeysView: collections.abc.KeysView,
#     typing.Mapping: collections.abc.Mapping,
#     typing.MappingView: collections.abc.MappingView,
#     typing.MutableMapping: collections.abc.MutableMapping,
#     typing.MutableSequence: collections.abc.MutableSequence,
#     typing.MutableSet: collections.abc.MutableSet,
#     typing.Sequence: collections.abc.Sequence,
#     typing.ValuesView: collections.abc.ValuesView,
#     typing.Coroutine: collections.abc.Coroutine,
#     typing.AsyncGenerator: collections.abc.AsyncGenerator,
#     typing.AsyncIterable: collections.abc.AsyncIterable,
#     typing.AsyncIterator: collections.abc.AsyncIterator,
#     typing.Awaitable: collections.abc.Awaitable,
#     typing.Generator: collections.abc.Generator,
#     typing.Iterable: collections.abc.Iterable,
#     typing.Iterator: collections.abc.Iterator,
#     typing.Callable: collections.abc.Callable,
#     typing.Hashable: collections.abc.Hashable,
#     typing.Reversible: collections.abc.Reversible,
#     typing.Sized: collections.abc.Sized,
#     # contextlib aliases
#     typing.ContextManager: contextlib.AbstractContextManager,
#     typing.AsyncContextManager: contextlib.AbstractAsyncContextManager,
# }


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


def normalize_annotation(
    annotation: TypeAnnotation, concretize: bool = False
) -> TypeAnnotation:
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
    """
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        return annotation
    if args:
        args = tuple(normalize_annotation(arg, concretize=concretize) for arg in args)
    if origin is Annotated:
        return args[0]
    if origin is Union:
        return _fold_or(args)
    if concretize:
        origin = ABSTRACT_TO_CONCRETE.get(origin, origin)
    if args:
        return origin[args]
    return origin
