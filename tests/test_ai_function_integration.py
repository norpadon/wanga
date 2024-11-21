from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import List, Literal, Optional

import pytest

from wanga.function import ai_function
from wanga.models.openai import OpenAIModel
from wanga.runtime import Runtime

model = OpenAIModel("gpt-4o-mini")


@pytest.fixture(scope="module")
def runtime():
    with Runtime(model) as rt:
        yield rt


def test_basic_ai_function(runtime):
    @ai_function()
    def greet(name: str) -> str:
        """
        [|system|]
        You are a friendly assistant.

        [|user|]
        Greet {{ name }} in a friendly manner.
        """
        raise NotImplementedError

    result = greet("Alice")
    assert isinstance(result, str)
    assert "Alice" in result


def test_ai_function_with_tools(runtime):
    def get_current_time() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @ai_function(tools=[get_current_time])
    def time_greeting(name: str) -> str:
        """
        [|system|]
        You are a helpful assistant who can tell the time.

        [|user|]
        Greet {{ name }} and tell them the current time.
        """
        raise NotImplementedError

    result = time_greeting("Bob")
    assert isinstance(result, str)
    assert "Bob" in result
    assert ":" in result  # Checking for time format


def test_ai_function_with_complex_return_type(runtime):
    from dataclasses import dataclass

    @dataclass
    class WeatherInfo:
        temperature: float
        conditions: str

    @ai_function()
    def get_weather(city: str) -> WeatherInfo:
        """
        [|system|]
        You are a weather information provider.

        [|user|]
        Provide the current weather for {{ city }}. Use realistic but made-up data.
        """
        raise NotImplementedError

    result = get_weather("New York")
    assert isinstance(result, WeatherInfo)
    assert isinstance(result.temperature, float)
    assert isinstance(result.conditions, str)


def test_ai_function_with_complex_prompt(runtime):
    @ai_function()
    def analyze_text(text: str, focus: str, word_limit: int) -> str:
        """
        [|system|]
        You are a text analysis expert.

        [|user|]
        Analyze the following text:

        {{ text }}

        Focus on aspects related to {{ focus }}.
        Provide your analysis in {{ word_limit }} words or less.
        """
        raise NotImplementedError

    result = analyze_text(text="The quick brown fox jumps over the lazy dog.", focus="animal behavior", word_limit=50)
    assert isinstance(result, str)
    assert len(result.split()) <= 55


def test_ai_function_with_multiple_tools(runtime):
    num_calls = 0

    def add(a: int, b: int) -> int:
        nonlocal num_calls
        num_calls += 1
        return a + b

    def multiply(a: int, b: int) -> int:
        nonlocal num_calls
        num_calls += 1
        return a * b

    @ai_function(tools=[add, multiply])
    def complex_calculation(x: int, y: int, z: int) -> int:
        """
        [|system|]
        You are a math assistant with addition and multiplication capabilities.

        [|user|]
        Perform the following calculation:
        1. Add {{ x }} and {{ y }}
        2. Multiply the result by {{ z }}
        """
        raise NotImplementedError

    result = complex_calculation(5, 3, 2)
    assert result == 16  # (5 + 3) * 2 = 16
    assert num_calls == 2  # Two tool functions were called


@dataclass
class Address:
    street: str
    city: str
    country: str
    postal_code: str


@dataclass
class Person:
    name: str
    age: int
    address: Address
    birth_date: date


def test_ai_function_with_nested_dataclasses(runtime):
    @ai_function()
    def generate_person_info() -> Person:
        """
        [|system|]
        You are an AI assistant capable of generating realistic person information.

        [|user|]
        Generate information for a fictional person, including their name, age, address, and birth date.
        """
        raise NotImplementedError

    result = generate_person_info()
    assert isinstance(result, Person)
    assert isinstance(result.address, Address)
    assert isinstance(result.birth_date, date)
    assert 0 <= result.age <= 120


JobStatus = Literal["Employed", "Unemployed", "Student"]


@dataclass
class WorkExperience:
    company: str
    position: str
    start_date: date
    end_date: Optional[date] = None


@dataclass
class ComplexPerson:
    name: str
    age: int
    status: JobStatus
    experiences: List[WorkExperience]
    last_update: datetime


def test_ai_function_with_complex_structures(runtime):
    @ai_function()
    def generate_complex_person() -> ComplexPerson:
        """
        [|system|]
        You are an AI assistant capable of generating detailed person profiles.

        [|user|]
        Generate a complex person profile with name, age, job status, work experiences, and last update time.
        Include at least two work experiences.
        """
        raise NotImplementedError

    result = generate_complex_person()
    assert isinstance(result, ComplexPerson)
    assert len(result.experiences) >= 2
    for exp in result.experiences:
        assert isinstance(exp, WorkExperience)
        assert isinstance(exp.start_date, date)
        assert exp.end_date is None or isinstance(exp.end_date, date)
    assert isinstance(result.last_update, datetime)


@dataclass
class TimeRange:
    start: time
    end: time
    duration: timedelta


def test_ai_function_with_time_structures(runtime):
    @ai_function()
    def generate_work_schedule() -> List[TimeRange]:
        """
        [|system|]
        You are an AI assistant capable of generating work schedules.

        [|user|]
        Generate a work schedule for a typical day, with at least 3 time ranges.
        Each time range should have a start time, end time, and duration.
        """
        raise NotImplementedError

    result = generate_work_schedule()
    assert isinstance(result, list)
    assert len(result) >= 3
    for time_range in result:
        assert isinstance(time_range, TimeRange)
        assert isinstance(time_range.start, time)
        assert isinstance(time_range.end, time)
        assert isinstance(time_range.duration, timedelta)
        assert time_range.duration == timedelta(
            hours=time_range.end.hour - time_range.start.hour,
            minutes=time_range.end.minute - time_range.start.minute,
        )
