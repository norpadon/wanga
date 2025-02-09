from importlib.util import find_spec

from .function import ai_function
from .models.openai import OpenAIModel
from .runtime import Runtime

__all__ = ["ai_function", "OpenAIModel", "Runtime"]

_PIL_INSTALLED = find_spec("Pillow") is None
_OPENAI_INSTALLED = find_spec("openai") is None
_ANTHROPIC_INSTALLED = find_spec("anthropic") is None
