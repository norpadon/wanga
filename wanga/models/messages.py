from pprint import pformat
from typing import TYPE_CHECKING, TypeAlias

from attrs import frozen

from wanga import _PIL_INSTALLED

from ..common import JSON

__all__ = [
    "ImageURL",
    "ImageContent",
    "Message",
    "UserMessage",
    "AssistantMessage",
    "ToolInvocation",
    "ToolMessage",
]


@frozen
class ImageURL:
    url: str


if TYPE_CHECKING or _PIL_INSTALLED:
    from PIL.Image import Image

    ImageContent: TypeAlias = Image | ImageURL
else:
    ImageContent: TypeAlias = ImageURL


_MESSAGE_TAG_OPEN = "[|"
_MESSAGE_TAG_CLOSE = "|]"

_CONTENT_BLOCK_TAG_OPEN = "<|"
_CONTENT_BLOCK_TAG_CLOSE = "|>"


def _pretty_print_message(role: str, name: str | None, content: str | list[str | ImageContent]) -> str:
    if name:
        role_str = f"{_MESSAGE_TAG_OPEN}{role} {name}{_MESSAGE_TAG_CLOSE}"
    else:
        role_str = f"{_MESSAGE_TAG_OPEN}{role}{_MESSAGE_TAG_CLOSE}"

    content_strings = []
    if isinstance(content, str):
        content = [content]
    for content_block in content:
        if isinstance(content_block, ImageURL):
            content_block = f'{_CONTENT_BLOCK_TAG_OPEN}image url="{content_block.url}"{_CONTENT_BLOCK_TAG_CLOSE}'
        if not isinstance(content_block, str):
            content_block = f"{_CONTENT_BLOCK_TAG_OPEN}image{_CONTENT_BLOCK_TAG_CLOSE}"
        content_strings.append(content_block)
    joined_content = "\n".join(content_strings)

    return f"{role_str}\n{joined_content}"


@frozen
class Message:
    pass


class SystemMessage(Message):
    name: str | None
    content: str

    def __str__(self) -> str:
        return _pretty_print_message("system", self.name, self.content)


@frozen
class UserMessage(Message):
    name: str | None
    content: str | list[str | ImageContent]

    def __str__(self) -> str:
        return _pretty_print_message("user", self.name, self.content)


@frozen
class ToolInvocation:
    invocation_id: str
    tool_name: str
    tool_args: JSON

    def __str__(self) -> str:
        pretty_json = pformat(self.tool_args, sort_dicts=True)
        call_header = (
            f'{_CONTENT_BLOCK_TAG_OPEN}call {self.tool_name} id="{self.invocation_id}"{_CONTENT_BLOCK_TAG_CLOSE}'
        )
        return f"{call_header}\n{pretty_json}"


@frozen
class AssistantMessage(Message):
    name: str | None
    content: str | None
    tool_invocations: list[ToolInvocation]

    def __str__(self) -> str:
        content_blocks = []
        if self.content:
            content_blocks.append(self.content)
        for tool_invocation in self.tool_invocations:
            content_blocks.append(str(tool_invocation))
        return "\n".join(content_blocks)


@frozen
class ToolMessage:
    invocation_id: str
    tool_result: str | list[str | ImageContent]

    def __str__(self) -> str:
        return _pretty_print_message("tool", f"id={self.invocation_id}", self.tool_result)
