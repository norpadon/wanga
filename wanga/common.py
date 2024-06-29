from typing import TypeAlias

JSON: TypeAlias = str | float | int | bool | None | dict[str, "JSON"] | list["JSON"]
