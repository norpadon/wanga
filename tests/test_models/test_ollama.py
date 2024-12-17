import asyncio
from textwrap import dedent

import pytest

from wanga.models.messages import parse_messages
from wanga.models.ollama import OllamaModel


@pytest.fixture
def model():
    return OllamaModel("llama3.2")


@pytest.fixture
def vision_model():
    return OllamaModel("llava")


def test_reply(model):
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


# Llava is currently broken :(
# def test_vision(vision_model):
#     messages = parse_messages(
#         "How many kittens are in this image?"
#         '<|image url="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWHPE0dCs93yAjfxnT2IOR-lbNhvur5FlmkQ&s"|>'
#     )
#     response = vision_model.reply(messages)
#     assert "four" in response.response_options[0].message.content.lower()  # type: ignore
