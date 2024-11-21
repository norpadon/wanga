import asyncio
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Request
from openai import APIConnectionError, RateLimitError

from wanga.models.messages import parse_messages
from wanga.models.model import FinishReason, ModelResponse, ToolParams
from wanga.models.model import RateLimitError as WangaRateLimitError
from wanga.models.openai import OpenAIModel
from wanga.schema import DEFAULT_SCHEMA_EXTRACTOR


def test_reply():
    model = OpenAIModel("gpt-4o-mini")
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
    assert OpenAIModel("gpt-4-turbo").context_length == 128000
    assert OpenAIModel("gpt-4").context_length == 8192


def test_num_tokens():
    model = OpenAIModel("gpt-4o-mini")
    prompt = r"""
    [|system|]
    You are a helpful assistant.
    [|user|]
    2 + 2 = ?
    """

    def tool(x: int, y: str):
        pass

    tool_schema = DEFAULT_SCHEMA_EXTRACTOR.extract_schema(tool)
    prompt = dedent(prompt.removeprefix("\n"))
    messages = parse_messages(prompt)
    tools = ToolParams(tools=[tool_schema])
    assert abs(model.estimate_num_tokens(messages, tools) - model.calculate_num_tokens(messages, tools)) < 2


@pytest.fixture
def model():
    return OpenAIModel("gpt-4o-mini", num_retries=2, retry_on_request_limit=True)


@pytest.fixture
def vision_model():
    return OpenAIModel("gpt-4o", num_retries=2, retry_on_request_limit=True)


def test_retry_on_rate_limit(model):
    with patch.object(model._client.chat.completions, "create") as mock_create:
        mock_create.side_effect = [
            RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body={"error": {"message": "Rate limit exceeded"}},
            ),
            RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body={"error": {"message": "Rate limit exceeded"}},
            ),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="4"), finish_reason="stop", index=0)],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            ),
        ]

        messages = parse_messages("2 + 2 = ?")
        response = model.reply(messages)

        assert mock_create.call_count == 3
        assert isinstance(response, ModelResponse)
        assert len(response.response_options) == 1
        assert response.response_options[0].finish_reason == FinishReason.STOP
        assert "4" in response.response_options[0].message.content  # type: ignore
        assert response.usage.prompt_tokens == 10
        assert response.usage.response_tokens == 5


def test_retry_on_service_unavailable(model):
    with patch.object(model._client.chat.completions, "create") as mock_create:
        mock_create.side_effect = [
            APIConnectionError(message="Service unavailable", request=Request("get", "https://api.openai.com")),
            APIConnectionError(message="Service unavailable", request=Request("get", "https://api.openai.com")),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="4"), finish_reason="stop", index=0)],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            ),
        ]

        messages = parse_messages("2 + 2 = ?")
        response = model.reply(messages)

        assert mock_create.call_count == 3
        assert isinstance(response, ModelResponse)
        assert len(response.response_options) == 1
        assert response.response_options[0].finish_reason == FinishReason.STOP
        assert "4" in response.response_options[0].message.content  # type: ignore
        assert response.usage.prompt_tokens == 10
        assert response.usage.response_tokens == 5


def test_no_retry_on_other_errors(model):
    with patch.object(model._client.chat.completions, "create") as mock_create:
        mock_create.side_effect = ValueError("Some other error")

        messages = parse_messages("2 + 2 = ?")
        with pytest.raises(ValueError):
            model.reply(messages)

        assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_async_retry_on_rate_limit(model):
    mock_response = MagicMock(
        choices=[MagicMock(message=MagicMock(content="4"), finish_reason="stop", index=0)],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
    )

    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body={"error": {"message": "Rate limit exceeded"}},
            )
        return mock_response

    with patch.object(model._async_client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = side_effect

        messages = parse_messages("2 + 2 = ?")
        response = await model.reply_async(messages)

        assert call_count == 3
        assert isinstance(response, ModelResponse)
        assert len(response.response_options) == 1
        assert response.response_options[0].finish_reason == FinishReason.STOP
        assert "4" in response.response_options[0].message.content  # type: ignore
        assert response.usage.prompt_tokens == 10
        assert response.usage.response_tokens == 5


def test_no_retry_when_disabled(model):
    model._retry_on_request_limit = False
    with patch.object(model._client.chat.completions, "create") as mock_create:
        mock_create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}},
        )

        messages = parse_messages("2 + 2 = ?")
        with pytest.raises(WangaRateLimitError):
            model.reply(messages)

        assert mock_create.call_count == 1


def test_vision(vision_model):
    messages = parse_messages(
        "How many kittens are in this image?"
        '<|image url="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWHPE0dCs93yAjfxnT2IOR-lbNhvur5FlmkQ&s"|>'
    )
    response = vision_model.reply(messages)
    assert "four" in response.response_options[0].message.content.lower()  # type: ignore
