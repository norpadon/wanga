import inspect
from datetime import date, datetime, time, timedelta
from typing import Callable, TypeAlias

from .schema import (
    CallableSchema,
    ObjectField,
    ObjectNode,
    PrimitiveNode,
    UndefinedNode,
)
from .utils import TypeAnnotation, strip_self

ExtractorFn: TypeAlias = Callable[[TypeAnnotation], CallableSchema | None]


__all__ = [
    "extract_datetime",
]


def _get_datetime_arg(name: str, required: bool = True) -> ObjectField:
    return ObjectField(
        name=name,
        schema=PrimitiveNode(primitive_type=int),
        required=required,
        hint=None,
    )


def extract_datetime(annotation: TypeAnnotation) -> CallableSchema | None:
    date_fields = ["year", "month", "day"]
    time_fields = ["hour", "minute", "second"]  # We deliberately omit microseconds

    delta_fields = ["days", "seconds"]  # We deliberately omit microseconds

    fields = []
    if annotation in [date, datetime]:
        fields.extend(_get_datetime_arg(name) for name in date_fields)
    if annotation in [datetime, time]:
        fields.extend(_get_datetime_arg(name, required=False) for name in time_fields)
    if annotation is timedelta:
        fields.extend(_get_datetime_arg(name, required=False) for name in delta_fields)

    if not fields:
        return None

    return CallableSchema(
        call_schema=ObjectNode(
            constructor_fn=annotation,
            constructor_signature=strip_self(inspect.signature(annotation.__init__)),
            name=annotation.__name__,
            fields=fields,
            hint=None,
        ),
        return_schema=UndefinedNode(original_annotation=None),
        long_description=None,
    )
