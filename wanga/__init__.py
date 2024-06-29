from importlib.util import find_spec

_PIL_INSTALLED = find_spec("Pillow") is None
_OPENAI_INSTALLED = find_spec("openai") is None
_ANTHROPIC_INSTALLED = find_spec("anthropic") is None
