import pytest

from wanga.function import ai_function
from wanga.models import (
    AssistantMessage,
    FinishReason,
    GenerationParams,
    Message,
    Model,
    ModelResponse,
    ResponseOption,
    ToolInvocation,
    ToolParams,
    UsageStats,
)
from wanga.runtime import Runtime
from wanga.schema import SchemaValidationError


class MockModel(Model):
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        self._name = "mock-model"

    def reply(
        self,
        messages: list[Message],
        tools: ToolParams = ToolParams(),
        params: GenerationParams = GenerationParams(),
        num_options: int = 1,
        user_id: str | None = None,
    ) -> ModelResponse:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = self.responses[-1]  # Return the last response if we've run out
        self.call_count += 1
        return response

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def context_length(self) -> int:
        return 1000

    def estimate_num_tokens(self, messages: list[Message], tools: ToolParams) -> int:
        return 100

    def calculate_num_tokens(self, messages: list[Message], tools: ToolParams) -> int:
        return 100


def test_basic_ai_function_execution():
    mock_response = ModelResponse(
        response_options=[
            ResponseOption(
                message=AssistantMessage(
                    content=None,
                    tool_invocations=[
                        ToolInvocation(invocation_id="1", name="submit_response", arguments={"response": 3})
                    ],
                ),
                finish_reason=FinishReason.TOOL_CALL,
            )
        ],
        usage=UsageStats(prompt_tokens=10, response_tokens=5),
    )
    mock_model = MockModel([mock_response])

    with Runtime(mock_model):

        @ai_function()
        def add_numbers(a: int, b: int) -> int:
            """
            [|system|]
            You are a helpful math assistant.

            [|user|]
            Add {{ a }} and {{ b }}.

            [|assistant|]
            The sum of {{ a }} and {{ b }} is {{ a + b }}.
            """
            raise NotImplementedError

        result = add_numbers(1, 2)

    assert result == 3
    assert mock_model.call_count == 1


def test_ai_function_with_tools():
    mock_responses = [
        ModelResponse(
            response_options=[
                ResponseOption(
                    message=AssistantMessage(
                        content=None,
                        tool_invocations=[
                            ToolInvocation(invocation_id="1", name="multiply", arguments={"x": 3, "y": 2})
                        ],
                    ),
                    finish_reason=FinishReason.TOOL_CALL,
                )
            ],
            usage=UsageStats(prompt_tokens=15, response_tokens=10),
        ),
        ModelResponse(
            response_options=[
                ResponseOption(
                    message=AssistantMessage(
                        content=None,
                        tool_invocations=[
                            ToolInvocation(invocation_id="2", name="submit_response", arguments={"response": 6})
                        ],
                    ),
                    finish_reason=FinishReason.TOOL_CALL,
                )
            ],
            usage=UsageStats(prompt_tokens=20, response_tokens=5),
        ),
    ]
    mock_model = MockModel(mock_responses)

    def multiply(x: int, y: int) -> int:
        return x * y

    with Runtime(mock_model):

        @ai_function(tools=[multiply])
        def complex_math(a: int, b: int) -> int:
            """
            [|system|]
            You are a math assistant capable of addition and multiplication.

            [|user|]
            Add {{ a }} and {{ b }}, then multiply the result by 2.
            """
            raise NotImplementedError

        result = complex_math(1, 2)

    assert result == 6
    assert mock_model.call_count == 2


def test_ai_function_error_handling():
    mock_responses = [
        ModelResponse(
            response_options=[
                ResponseOption(
                    message=AssistantMessage(
                        content=None,
                        tool_invocations=[
                            ToolInvocation(
                                invocation_id="1", name="submit_response", arguments={"response": "Invalid"}
                            )
                        ],
                    ),
                    finish_reason=FinishReason.TOOL_CALL,
                )
            ],
            usage=UsageStats(prompt_tokens=10, response_tokens=5),
        )
    ] * 4  # Provide 4 invalid responses

    mock_model = MockModel(mock_responses)

    with Runtime(mock_model):

        @ai_function(max_retries_on_invalid_output=3)
        def error_prone_function() -> int:
            """
            [|system|]
            You are a helpful assistant.

            [|user|]
            Return the number 42.
            """
            raise NotImplementedError

        with pytest.raises(SchemaValidationError):
            error_prone_function()

    assert mock_model.call_count == 4  # Initial call + 3 retries


def test_ai_function_preferred_models():
    mock_model_1 = MockModel(
        [
            ModelResponse(
                response_options=[
                    ResponseOption(
                        message=AssistantMessage(
                            content=None,
                            tool_invocations=[
                                ToolInvocation(invocation_id="1", name="submit_response", arguments={"response": 42})
                            ],
                        ),
                        finish_reason=FinishReason.TOOL_CALL,
                    )
                ],
                usage=UsageStats(prompt_tokens=10, response_tokens=5),
            )
        ]
        * 2
    )  # Provide 2 responses to ensure we don't run out
    mock_model_1.name = "preferred-model"

    mock_model_2 = MockModel([])
    mock_model_2.name = "fallback-model"

    with Runtime([mock_model_1, mock_model_2]):

        @ai_function(preferred_models=["preferred-model"])
        def preferred_function() -> int:
            """
            [|system|]
            You are a helpful assistant.

            [|user|]
            Return the number 42.
            """
            raise NotImplementedError

        result = preferred_function()

    assert result == 42
    assert mock_model_1.call_count == 1
    assert mock_model_2.call_count == 0
