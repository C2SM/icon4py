import functools
import inspect

import pytest

from icon4py.model.common.utils import profiling


@pytest.fixture
def recorded_ranges(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    ranges: list[str] = []

    class FakeTimeRange:
        # Reusable like CuPy's `time_range`: a decorator pushes the same instance on every call.
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            ranges.append(self.name)

        def __exit__(self, *exc_info: object) -> None:
            pass

    monkeypatch.setattr(profiling.gpu_profiler, "time_range", FakeTimeRange)
    return ranges


def test_annotate_records_range_when_enabled(monkeypatch, recorded_ranges):
    monkeypatch.setattr(profiling, "ENABLED", True)

    with profiling.annotate("outer"), profiling.annotate("inner"):
        pass

    assert recorded_ranges == ["outer", "inner"]


def test_annotate_is_no_op_when_disabled(monkeypatch, recorded_ranges):
    monkeypatch.setattr(profiling, "ENABLED", False)

    with profiling.annotate("ignored"):
        pass

    assert recorded_ranges == []


@pytest.mark.parametrize("enabled", [True, False])
def test_annotate_requires_name_as_context_manager(monkeypatch, recorded_ranges, enabled):
    monkeypatch.setattr(profiling, "ENABLED", enabled)

    with pytest.raises(ValueError, match="needs a name"), profiling.annotate():
        pass


def test_decorator_names_range_after_qualname(monkeypatch, recorded_ranges):
    monkeypatch.setattr(profiling, "ENABLED", True)

    @profiling.annotate()
    def compute() -> None:
        pass

    compute()
    compute()

    assert recorded_ranges == ["test_decorator_names_range_after_qualname.<locals>.compute"] * 2


def test_decorator_uses_explicit_name(monkeypatch, recorded_ranges):
    monkeypatch.setattr(profiling, "ENABLED", True)

    @profiling.annotate("custom")
    def compute() -> None:
        pass

    compute()

    assert recorded_ranges == ["custom"]


def test_decorator_returns_function_unchanged_when_disabled(monkeypatch):
    monkeypatch.setattr(profiling, "ENABLED", False)

    def compute() -> None:
        pass

    assert profiling.annotate()(compute) is compute


def test_decorator_preserves_arguments_result_and_signature(monkeypatch, recorded_ranges):
    monkeypatch.setattr(profiling, "ENABLED", True)

    def scale(value: float, *, factor: float = 2.0) -> float:
        return value * factor

    annotated = profiling.annotate()(scale)

    assert annotated is not scale
    assert annotated(3.0, factor=4.0) == 12.0
    assert inspect.signature(annotated) == inspect.signature(scale)


def test_decorator_keeps_singledispatch_register(monkeypatch, recorded_ranges):
    monkeypatch.setattr(profiling, "ENABLED", True)

    @profiling.annotate("describe")
    @functools.singledispatch
    def describe(value: object) -> str:
        return "object"

    @describe.register(int)
    def _(value: int) -> str:
        return "int"

    assert describe(1) == "int"
    assert describe("text") == "object"
    assert recorded_ranges == ["describe", "describe"]
