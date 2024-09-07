from dataclasses import dataclass

import pytest
from jinja2 import Template

from wanga.common import JSON
from wanga.function import AIFunction, ai_function
from wanga.models import GenerationParams
from wanga.schema import CallableSchema
from wanga.schema.schema import SchemaValidationError


def test_basic_ai_function():
    @ai_function()
    def add_numbers(a: int, b: int) -> int:
        """
        [|system|]
        You are a helpful math assistant.

        [|user|]
        Please add the following numbers: {{ a }} and {{ b }}
        """
        raise

    assert isinstance(add_numbers.__ai_function, AIFunction)  # type: ignore
    assert list(add_numbers.__ai_function.signature.parameters.keys()) == ["a", "b"]  # type: ignore
    assert add_numbers.__ai_function.return_schema.call_schema.name == "submit_response"  # type: ignore
    assert isinstance(add_numbers.__ai_function.prompt_template, Template)  # type: ignore


def test_ai_function_with_tools():
    def multiply(x: int, y: int) -> int:
        return x * y

    @ai_function(tools=[multiply])
    def complex_math(a: int, b: int) -> int:
        """
        [|system|]
        You are a math assistant capable of addition and multiplication.

        [|user|]
        I need help with a math problem. First, add {{ a }} and {{ b }}, then multiply the result by 2.
        """
        raise

    assert isinstance(complex_math.__ai_function, AIFunction)  # type: ignore
    assert len(complex_math.__ai_function.tools) == 1  # type: ignore
    assert isinstance(complex_math.__ai_function.tools[0], CallableSchema)  # type: ignore
    assert complex_math.__ai_function.tools[0].call_schema.name == "multiply"  # type: ignore


def test_ai_function_with_preferred_models():
    @ai_function(preferred_models=["gpt-4", "claude-v1"])
    def simple_task(task: str) -> str:
        """
        [|system|]
        You are a helpful assistant.

        [|user|]
        Please help me with the following task: {{ task }}
        """
        raise

    assert isinstance(simple_task.__ai_function, AIFunction)  # type: ignore
    assert simple_task.__ai_function.preferred_models == ["gpt-4", "claude-v1"]  # type: ignore


def test_ai_function_with_generation_params():
    @ai_function(generation_params=GenerationParams(max_tokens=100, temperature=0.7))
    def creative_writing(topic: str) -> str:
        """
        [|system|]
        You are a creative writing assistant.

        [|user|]
        Write a short story about {{ topic }} in exactly 3 sentences.
        """
        raise

    assert isinstance(creative_writing.__ai_function, AIFunction)  # type: ignore
    assert creative_writing.__ai_function.generation_params.max_tokens == 100  # type: ignore
    assert creative_writing.__ai_function.generation_params.temperature == 0.7  # type: ignore


def test_ai_function_missing_return_annotation():
    with pytest.raises(ValueError, match="Function must have a concrete return type annotation."):

        @ai_function()
        def missing_return():
            """
            [|system|]
            You are a helpful assistant.
            """
            raise


def test_ai_function_missing_docstring():
    with pytest.raises(ValueError, match="Prompt is missing."):

        @ai_function()
        def missing_docstring() -> str:
            raise


def test_ai_function_invalid_template():
    with pytest.raises(ValueError, match="Variable invalid_var is not a parameter of the function."):

        @ai_function()
        def invalid_template(name: str) -> str:
            """
            [|system|]
            You are a helpful assistant.

            [|user|]
            Please greet {{ invalid_var }}
            """
            raise


def test_ai_function_return_schema():
    @ai_function()
    def greet(name: str) -> str:
        """
        [|system|]
        You are a friendly greeter.

        [|user|]
        Please greet {{ name }}.
        """
        raise NotImplementedError

    ai_func = greet.__ai_function  # type: ignore
    assert isinstance(ai_func, AIFunction)
    assert ai_func.return_schema is None  # For string return type, return_schema is None


def test_ai_function_eval_return_schema():
    @dataclass
    class RectangleArea:
        area: float
        perimeter: float

    @ai_function()
    def calculate_area(length: float, width: float) -> RectangleArea:
        """
        [|system|]
        You are a helpful assistant that calculates the area of rectangles.

        [|user|]
        Please calculate the area of a rectangle with length {{ length }} and width {{ width }}.
        """
        raise NotImplementedError

    ai_func = calculate_area.__ai_function  # type: ignore
    # Test with valid input
    valid_input: JSON = {"response": {"area": 50.0, "perimeter": 30.0}}
    result = ai_func.return_schema.eval(valid_input)
    assert isinstance(result, RectangleArea)
    assert result.area == 50.0
    assert result.perimeter == 30.0

    # Test with invalid input (missing field)
    with pytest.raises(SchemaValidationError):
        ai_func.return_schema.eval({"response": {"area": 50.0}})

    # Test with invalid input (wrong type)
    with pytest.raises(SchemaValidationError):
        ai_func.return_schema.eval({"response": {"area": "50.0", "perimeter": 30.0}})

    # Test with invalid input (extra field)
    with pytest.raises(SchemaValidationError):
        ai_func.return_schema.eval({"response": {"area": 50.0, "perimeter": 30.0, "extra": 10}})
