"""
Profiler range markers, recorded through GT4Py's GPU profiler.

GT4Py records the ranges with CuPy, which emits NVTX ranges on NVIDIA GPUs and
ROCTx ranges on AMD GPUs, so they show up on the timeline of `nsys` or
`rocprofv3` without icon4py depending on either marker library.

`annotate` records every call of a function when used as a decorator, and the
enclosed block when used as a context manager::

    @profiling.annotate()
    def run_driver(...): ...

    with profiling.annotate("time_loop"):
        ...

Markers are disabled by default and are enabled by setting the
`ICON4PY_PROFILE_MARKERS` environment variable to a true value (e.g. `1`). The
variable is read once, when this module is imported. A decorator checks it when
it is applied: with markers disabled it returns the function unchanged.

To record the markers, run under a profiler with marker tracing enabled::

    ICON4PY_PROFILE_MARKERS=1 nsys profile --trace=nvtx,cuda <command>
    ICON4PY_PROFILE_MARKERS=1 rocprofv3 --marker-trace -- <command>

Note:
    Ranges are pushed and popped on a per-thread stack, so they must nest.
    They are recorded on the host: around asynchronous GPU work they measure
    the submission, not the device-side execution.
"""

import contextlib
import functools
import types
from collections.abc import Callable
from typing import Any, cast

from gt4py.next.instrumentation import gpu_profiler

from icon4py.model.common.utils import env


ENABLED: bool = env.flag_to_bool("ICON4PY_PROFILE_MARKERS", False)
"""Whether ranges are recorded."""


class _Range(contextlib.AbstractContextManager[None]):
    """A profiler range, usable as a context manager or as a function decorator."""

    def __init__(self, name: str | None) -> None:
        self._name = name
        # Without a GPU, GT4Py's fallback `time_range` warns on construction, so it is
        # only created when markers are requested.
        self._time_range: contextlib.AbstractContextManager[Any] | None = (
            gpu_profiler.time_range(name) if ENABLED and name is not None else None
        )

    def __enter__(self) -> None:
        if self._name is None:
            raise ValueError("A profiler range used as a context manager needs a name.")
        if self._time_range is not None:
            self._time_range.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if self._time_range is not None:
            self._time_range.__exit__(exc_type, exc_value, traceback)

    def __call__[F: Callable[..., Any]](self, func: F) -> F:
        if not ENABLED:
            return func
        # A time range holds no per-call state, so one instance serves every call.
        time_range = (
            self._time_range
            if self._time_range is not None
            else gpu_profiler.time_range(func.__qualname__)
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with time_range:
                return func(*args, **kwargs)

        # `functools.wraps` also copies the attributes of `func`, e.g. `register` of a
        # `functools.singledispatch` function, so the wrapper keeps the type of `func`.
        return cast(F, wrapper)


def annotate(name: str | None = None) -> _Range:
    """
    Record a profiler range called `name`.

    As a decorator, the range covers every call of the function and `name`
    defaults to the function's `__qualname__`. As a context manager, the range
    covers the enclosed block and `name` is required.
    """
    return _Range(name)
