import json
import logging
import re
from collections.abc import Iterable
from typing import NamedTuple

from attrs import field, frozen

from ..common import JSON

__all__ = [
    "AssistantMessage",
    "ImageContent",
    "Message",
    "ToolInvocation",
    "ToolMessage",
    "UserMessage",
    "parse_messages",
]


class TagPair(NamedTuple):
    open: str
    close: str


_MESSAGE_TAGS = TagPair(r"[|", r"|]")
_CONTENT_BLOCK_TAGS = TagPair(r"<|", r"|>")


_logger = logging.getLogger(__name__)


def _format_header(_tags: TagPair, _name: str, /, **kwargs: str) -> str:
    param_str = "".join(f' {key}="{value}"' for key, value in kwargs.items())
    return f"{_tags.open}{_name}{param_str}{_tags.close}"


@frozen
class ImageContent:
    r"""Image embedded in a message. Can either be a URL, or a base64-encoded file."""

    url: str | None = None
    base64: str | None = None

    def __attrs_post_init__(self):
        if self.url is None and self.base64 is None:
            raise ValueError("ImageURL must have either a URL or base64 data")
        if self.url is not None and self.base64 is not None:
            raise ValueError("ImageURL cannot have both a URL and base64 data")

    def __str__(self) -> str:
        kwargs = {}
        if self.url:
            kwargs["url"] = self.url
        if self.base64:
            kwargs["base64"] = self.base64
        return _format_header(_CONTENT_BLOCK_TAGS, "image", **kwargs)


@frozen
class Message:
    pass


@frozen
class SystemMessage(Message):
    content: str
    name: str | None = None

    def __str__(self) -> str:
        kwargs = {}
        if self.name:
            kwargs["name"] = self.name
        header = _format_header(_MESSAGE_TAGS, "system", **kwargs)
        return f"{header}\n{self.content}"


@frozen
class UserMessage(Message):
    content: str | list[str | ImageContent]
    name: str | None = None

    def __str__(self) -> str:
        kwargs = {}
        if self.name:
            kwargs["name"] = self.name
        header = _format_header(_MESSAGE_TAGS, "user", **kwargs)
        content = "".join(str(item) for item in self.content)
        return f"{header}\n{content}"


@frozen
class ToolInvocation:
    invocation_id: str
    name: str
    arguments: JSON

    def __str__(self) -> str:
        pretty_json = json.dumps(self.arguments, indent=4, sort_keys=True)
        header = _format_header(_CONTENT_BLOCK_TAGS, "tool", name=self.name, id=self.invocation_id)
        return f"{header}\n{pretty_json}"


@frozen
class AssistantMessage(Message):
    name: str | None = None
    content: str | None = None
    tool_invocations: list[ToolInvocation] = field(factory=list)

    def __str__(self) -> str:
        kwargs = {}
        if self.name:
            kwargs["name"] = self.name
        header = _format_header(_MESSAGE_TAGS, "assistant", **kwargs)

        content_blocks = []
        if self.content:
            content_blocks.append(self.content)
        for tool_invocation in self.tool_invocations:
            content_blocks.append(str(tool_invocation))
        joined_content = "\n".join(content_blocks)

        return f"{header}\n{joined_content}"


@frozen
class ToolMessage(Message):
    invocation_id: str
    content: str | list[str | ImageContent]

    def __str__(self) -> str:
        header = _format_header(_MESSAGE_TAGS, "tool", id=self.invocation_id)
        tool_result = self.content
        if not isinstance(self.content, list):
            tool_result = [self.content]
        content = "".join(str(item) for item in tool_result)
        return f"{header}\n{content}"


class ParsedHeader(NamedTuple):
    name: str
    params: dict[str, str]


class MessageSyntaxError(ValueError):
    pass


class HeaderRegexes(NamedTuple):
    tag_regex: re.Pattern
    full_regex: re.Pattern

    def parse(self, header_str: str) -> ParsedHeader:
        header_match = self.full_regex.match(header_str)
        if header_match is None:
            raise MessageSyntaxError(f"Invalid header: {header_str}.")
        name = header_match.group("name")
        params_str = header_match.group("params")
        params = {}
        for param_match in _PARAM_REGEX.finditer(params_str):
            params[param_match.group("param_name")] = param_match.group("param_value")
        return ParsedHeader(name, params)

    def split_text(self, text: str) -> Iterable[ParsedHeader | str]:
        for text_block in self.tag_regex.split(text):
            if self.tag_regex.match(text_block):
                yield self.parse(text_block)
            elif text_block:
                yield text_block


_URL_SPECIAL_SYMBOLS = re.escape(r"/:!#$%&'*+-.^_`|~?=")
_PARAM_KEY_SYMBOLS = r"[a-zA-Z0-9_\-]"
_PARAM_VALUE_SYMBOLS = f"[a-zA-Z0-9{_URL_SPECIAL_SYMBOLS}]"


_PARAM_REGEX_STR = f'(?P<param_name>{_PARAM_KEY_SYMBOLS}+) *= *["](?P<param_value>{_PARAM_VALUE_SYMBOLS}*)["]'
_PARAM_REGEX = re.compile(_PARAM_REGEX_STR)

_PARAMS_REGEX_STR = f"( +(?P<param>{_PARAM_REGEX_STR}))* *"
_HEADER_BODY_REGEX_STR = f" *(?P<name>{_PARAM_KEY_SYMBOLS}+)(?P<params>{_PARAMS_REGEX_STR})"


def _make_header_regex(tags: TagPair, inner_regex: str) -> HeaderRegexes:
    open_tag = re.escape(tags.open)
    close_tag = re.escape(tags.close)
    tag_regex = re.compile(f"({open_tag}.*{close_tag})")
    full_regex = re.compile(f"(^{open_tag}{inner_regex}{close_tag}$)")
    return HeaderRegexes(tag_regex, full_regex)


_MESSAGE_HEADER_REGEXES = _make_header_regex(_MESSAGE_TAGS, _HEADER_BODY_REGEX_STR)
_CONTENT_HEADER_REGEXES = _make_header_regex(_CONTENT_BLOCK_TAGS, _HEADER_BODY_REGEX_STR)


def _parse_image_content(header: ParsedHeader) -> ImageContent:
    assert header.name == "image"
    try:
        return ImageContent(**header.params)
    except ValueError as e:
        raise MessageSyntaxError(f"Invalid image parameters: {header.params}") from e


def _parse_tool_invocation(header: ParsedHeader, arg_text: str) -> ToolInvocation:
    assert header.name == "tool"
    try:
        kwargs = dict(header.params)
        kwargs["invocation_id"] = kwargs.pop("id")
        return ToolInvocation(**kwargs, arguments=json.loads(arg_text))
    except (ValueError, KeyError) as e:
        raise MessageSyntaxError(f"Invalid tool invocation parameters: {header}\n{arg_text}") from e


def _map_headers_to_content(blocks: Iterable[str | ParsedHeader]) -> Iterable[str | ImageContent]:
    for block in blocks:
        if isinstance(block, str):
            yield block
        else:
            yield _parse_image_content(block)


def _parse_message(header: ParsedHeader, message_str: str) -> Message:
    message_blocks = list(_CONTENT_HEADER_REGEXES.split_text(message_str))
    match header.name:
        case "system":
            if len(message_blocks) > 1:
                raise MessageSyntaxError(f"System message cannot contain anything other than text: {message_str}")
            return SystemMessage(name=header.params.get("name"), content=message_str)
        case "user":
            content = list(_map_headers_to_content(message_blocks))
            if len(content) == 1 and isinstance(content[0], str):
                content = content[0]
            return UserMessage(name=header.params.get("name"), content=content)
        case "assistant":
            if not message_blocks:
                raise MessageSyntaxError(f"No content in assistant message: {message_str}")
            if isinstance(message_blocks[0], ParsedHeader):
                content = None
            else:
                content = message_blocks.pop(0)
                assert isinstance(content, str)
            tool_invocations = []
            for block_header, arg_str in zip(message_blocks[::2], message_blocks[1::2]):  # type: ignore
                if not isinstance(block_header, ParsedHeader):
                    raise MessageSyntaxError(f"Invalid tool invocation header: {block_header}")
                if not isinstance(arg_str, str):
                    raise MessageSyntaxError(f"No arguments specified to tool invocation {block_header}")
                tool_invocations.append(_parse_tool_invocation(block_header, arg_str))
            if tool_invocations and content is not None:
                content = content.removesuffix("\n")
            return AssistantMessage(
                name=header.params.get("name"),
                content=content,
                tool_invocations=tool_invocations,
            )
        case "tool":
            content = list(_map_headers_to_content(message_blocks))
            return ToolMessage(invocation_id=header.params["id"], content=content)
        case _:
            raise MessageSyntaxError(f"Invalid message type: {header.name}")


def parse_messages(chat_str: str) -> list[Message]:
    blocks = list(_MESSAGE_HEADER_REGEXES.split_text(chat_str))
    if not blocks:
        return []
    if not isinstance(blocks[0], ParsedHeader):
        blocks = [ParsedHeader("user", {})] + blocks
    messages = []
    for header, message_str in zip(blocks[::2], blocks[1::2]):
        if not isinstance(header, ParsedHeader):
            raise MessageSyntaxError(f"Invalid message header: {header}")
        if not isinstance(message_str, str):
            raise MessageSyntaxError(f"No content for message: {header}")
        message_str = message_str.removeprefix("\n").removesuffix("\n")
        if not message_str.strip():
            _logger.warning(
                f"Message doesn't contain non-whitespace symbols: {header}."
                "Check newlines at the begginning and the end of the prompt."
            )
        messages.append(_parse_message(header, message_str))
    return messages
