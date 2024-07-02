from textwrap import dedent

from wanga.models.messages import (
    AssistantMessage,
    ImageContent,
    SystemMessage,
    ToolInvocation,
    ToolMessage,
    UserMessage,
    parse_messages,
)


def test_consistency():
    messages = [
        SystemMessage(content="Be a helpful assistant!\n"),
        UserMessage(name="Alice", content="Hello, world!"),
        AssistantMessage(
            name="Bob",
            content="Here's a helpful image!",
            tool_invocations=[
                ToolInvocation(
                    invocation_id="abc123",
                    name="paste_image",
                    arguments={"url": "https://example.com/image.png"},
                )
            ],
        ),
        ToolMessage(
            invocation_id="abc123",
            content=[
                "Here is the image you requested:\n\n",
                ImageContent(url="https://example.com/image.png"),
            ],
        ),
        UserMessage(name="Alice", content="Thanks for the image!"),
    ]

    chat_str = "\n".join(str(message) for message in messages)
    parsed_messages = parse_messages(chat_str)
    assert parsed_messages == messages

    chat_str = r"""
    [|user|]
    Hi! Here is the pic<|image url="https://example.com/image.png"|>
    [|assistant|]
    It is beautiful!
    [|user|]
    I love you!
    [|assistant|]
    I love you too!
    """
    chat_str = dedent(chat_str.removeprefix("\n"))

    parsed_messages = parse_messages(chat_str)
    assert "\n".join(str(message) for message in parsed_messages) == chat_str.strip()


def test_num_blocks():
    chat_str = r"""
    [|system|]
    You are a helpful assistant.
    [|user|]
    2 + 2 = ?
    """
    chat_str = dedent(chat_str.removeprefix("\n"))
    parsed_messages = parse_messages(chat_str)
    system_message = parsed_messages[0]
    user_message = parsed_messages[1]
    assert isinstance(system_message, SystemMessage)
    assert isinstance(system_message.content, str)
    assert isinstance(user_message, UserMessage)
    assert isinstance(user_message.content, str)
