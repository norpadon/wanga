from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, TypeAlias

__all__ = [
    "JSON",
    "openai",
    "anthropic",
]

JSON: TypeAlias = str | float | int | bool | None | dict[str, "JSON"] | list["JSON"]


class LazyModule(ModuleType):
    def __init__(self, name: str, extra: str, self_name: str):
        super().__init__(name)
        self.extra = extra
        self.self_name = self_name

    def __getattr__(self, name: str):
        try:
            module = import_module(self.__name__)
        except ImportError:
            module = import_module(
                f"Module {self.__name__} not installed, install {self.extra}"
                f"capability using `pip install {self.self_name}[{self.extra}]`"
            )
        return getattr(module, name)


if TYPE_CHECKING:
    import anthropic
    import openai
    import openai_function_tokens
else:
    openai: ModuleType = LazyModule("openai", "openai", "wanga")
    openai_function_tokens = LazyModule("openai_function_tokens", "openai", "wanga")
    anthropic: ModuleType = LazyModule("anthropic", "anthropic", "wanga")
