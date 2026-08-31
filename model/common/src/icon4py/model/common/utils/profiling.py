# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Profiler range markers backed by NVTX (NVIDIA) or ROCTx (AMD).

Ranges emitted through this module show up on the timeline of `nsys` or `rocprofv3`,
which makes it possible to attribute time to named sections of the Python code.

Markers are disabled by default and are enabled through the `ICON4PY_PROFILE_MARKERS`
environment variable:

- `off` (default): all markers are no-ops.
- `auto`: use `roctx` if it is importable, otherwise `nvtx`, otherwise no-ops.
- `nvtx`: use `nvtx`, raise if it is not importable.
- `roctx`: use `roctx`, raise if it is not importable.

An explicitly requested backend raises instead of falling back to no-ops, because a
profiling run that silently records nothing is worse than one that fails immediately.
The backend is resolved once, when this module is imported.

The marker libraries are deliberately not declared as dependencies of `icon4py-common`:

- NVIDIA: `pip install nvtx` (https://pypi.org/project/nvtx).
- AMD: `roctx` is not on PyPI. It is built and installed by ROCprofiler-SDK into
  `<rocm-install>/lib/pythonX.Y/site-packages` and has to be put on `PYTHONPATH`
  explicitly. It is only built for the Python versions selected at ROCm build time,
  so it may not be available for the Python version used here.

To record the markers, run under a profiler with marker tracing enabled::

    ICON4PY_PROFILE_MARKERS=auto nsys profile --trace=nvtx,cuda <command>
    ICON4PY_PROFILE_MARKERS=auto rocprofv3 --marker-trace -- <command>

Note:
    Ranges are recorded on the host. Around asynchronous work (e.g. a halo exchange
    scheduled on a stream) they measure the submission, not the device-side execution.
"""

from __future__ import annotations

import contextlib
import enum
import logging
from collections.abc import Callable, Iterator
from typing import Any

from icon4py.model.common.utils import env


log = logging.getLogger(__name__)

MARKERS_ENV_VAR = "ICON4PY_PROFILE_MARKERS"

RangeHandle = Any
"""Opaque handle of an open range: a `tuple[int, int]` for NVTX, an `int` for ROCTx."""


class MarkerBackend(enum.StrEnum):
    OFF = "off"
    AUTO = "auto"
    NVTX = "nvtx"
    ROCTX = "roctx"


_StartRange = Callable[[str], RangeHandle]
_EndRange = Callable[[RangeHandle], None]


def _no_op_start_range(name: str) -> RangeHandle:
    return None


def _no_op_end_range(handle: RangeHandle) -> None:
    pass


def _nvtx_markers() -> tuple[_StartRange, _EndRange]:
    # Imported lazily so that a run with markers disabled does not load the library.
    import nvtx  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level]

    return nvtx.start_range, nvtx.end_range


def _roctx_markers() -> tuple[_StartRange, _EndRange]:
    # Imported lazily so that a run with markers disabled does not load the library.
    import roctx  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level]

    return roctx.rangeStart, roctx.rangeStop


_MARKER_IMPORTS: dict[MarkerBackend, Callable[[], tuple[_StartRange, _EndRange]]] = {
    MarkerBackend.NVTX: _nvtx_markers,
    MarkerBackend.ROCTX: _roctx_markers,
}


def _select_markers(backend: MarkerBackend) -> tuple[_StartRange, _EndRange]:
    """Resolve `backend` into the pair of functions opening and closing a range."""
    if backend is MarkerBackend.OFF:
        return _no_op_start_range, _no_op_end_range

    if backend is MarkerBackend.AUTO:
        # `roctx` only exists as part of a ROCm installation while `nvtx` installs
        # anywhere (it needs no CUDA), so the former is the stronger vendor hint.
        for candidate in (MarkerBackend.ROCTX, MarkerBackend.NVTX):
            try:
                return _MARKER_IMPORTS[candidate]()
            except ImportError:
                continue
        log.warning(
            f"Neither 'roctx' nor 'nvtx' is importable, profiling markers are disabled "
            f"('{MARKERS_ENV_VAR}={backend}')."
        )
        return _no_op_start_range, _no_op_end_range

    try:
        return _MARKER_IMPORTS[backend]()
    except ImportError as err:
        raise RuntimeError(
            f"'{MARKERS_ENV_VAR}={backend}' requires the '{backend}' module, "
            f"which is not importable: {err}."
        ) from err


BACKEND: MarkerBackend = env.str_enum(MARKERS_ENV_VAR, MarkerBackend, MarkerBackend.OFF)
"""Marker backend, resolved from the environment when this module is imported."""

# `start_range(name)` opens a range and returns its handle, `end_range(handle)` closes it.
# Handle based ranges (instead of push/pop) are the common denominator of the two
# libraries and allow a range to span two different functions.
start_range, end_range = _select_markers(BACKEND)

ENABLED: bool = start_range is not _no_op_start_range
"""Whether ranges are actually recorded."""


@contextlib.contextmanager
def _annotated_range(name: str) -> Iterator[str]:
    handle = start_range(name)
    try:
        yield name
    finally:
        end_range(handle)


annotate: Callable[[str], contextlib.AbstractContextManager[str]] = (
    _annotated_range if ENABLED else contextlib.nullcontext
)
"""Context manager recording the enclosed block as a range called `name`."""
