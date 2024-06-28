from inspect import Signature
from typing import Any, TypeAlias

__all__ = [
    "TypeAnnotation",
]

# Python doesn't have a way to specify a type annotation for type annotations,
# so we use `Any` as a placeholder.
TypeAnnotation: TypeAlias = Any


def strip_self(signature: Signature) -> Signature:
    return signature.replace(parameters=[param for name, param in signature.parameters.items() if name != "self"])
