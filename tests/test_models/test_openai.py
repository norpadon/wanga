import asyncio
from textwrap import dedent

from wanga.models.messages import parse_messages
from wanga.models.model import ToolParams
from wanga.models.openai import OpenaAIModel
from wanga.schema import default_schema_extractor


def test_reply():
    model = OpenaAIModel("gpt-3.5-turbo")
    prompt = r"""
    [|system|]
    You are a helpful assistant.
    [|user|]
    2 + 2 = ?
    """
    prompt = dedent(prompt.removeprefix("\n"))

    messages = parse_messages(prompt)
    response = model.reply(messages)

    response_text = response.response_options[0].message.content
    assert isinstance(response_text, str)
    assert "4" in response_text

    async_response_text = asyncio.run(model.reply_async(messages), debug=True).response_options[0].message.content
    assert isinstance(async_response_text, str)
    assert "4" in async_response_text


def test_context_size():
    assert OpenaAIModel("gpt-4-turbo").context_length == 128000
    assert OpenaAIModel("gpt-4").context_length == 8192


def test_num_tokens():
    model = OpenaAIModel("gpt-3.5-turbo")
    prompt = r"""
    [|system|]
    You are a helpful assistant.
    [|user|]
    2 + 2 = ?
    """

    def tool(x: int, y: str):
        pass

    tool_schema = default_schema_extractor.extract_schema(tool)
    prompt = dedent(prompt.removeprefix("\n"))
    messages = parse_messages(prompt)
    tools = ToolParams(tools=[tool_schema])
    assert abs(model.estimate_num_tokens(messages, tools) - model.calculate_num_tokens(messages, tools)) < 2
