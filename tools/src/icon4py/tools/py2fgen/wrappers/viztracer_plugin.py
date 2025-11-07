# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
VizTracer plugin for PY2FGEN generated bindings.

To enable this plugin
- install `viztracer` ('pip install viztracer') or `icon4py-tools[profiling]`
- set the environment variable `PY2FGEN_EXTRA_CALLABLES=icon4py.tools.py2fgen.wrappers.viztracer_plugin:init`
- set the environment variable
   - `ICON4PY_TRACING_RANGE` in the format 'start:stop' to define the range of calls to be traced, and
   - `ICON4PY_TRACING_NAMES` to specify the names of the functions to be traced (comma-separated)
  Note that the calling range is global, i.e., the tracer will trace all functions in the specified range.
"""

import dataclasses
import os
from collections.abc import Callable

import viztracer  # type: ignore[import-not-found]

from icon4py.tools.py2fgen import runtime_config
from icon4py.tools.py2fgen.wrappers import grid_wrapper


@dataclasses.dataclass
class _Tracer:
    start: int
    stop: int
    _tracer: viztracer.VizTracer = dataclasses.field(default_factory=viztracer.VizTracer)
    _counter: int = 0

    def enter(self) -> None:
        if self.start <= self._counter < self.stop:
            self._tracer.start()
        self._counter += 1

    def exit(self) -> None:
        if self.start < self._counter <= self.stop:
            # "flush_as_finish" to avoid incomplete calls to `disable` which would visualize as nested calls
            self._tracer.stop(stop_option="flush_as_finish")
        if self._counter == self.stop:
            assert grid_wrapper.grid_state is not None
            rank = (
                ""
                if grid_wrapper.grid_state.exchange_runtime.get_size() == 1
                else f"_rank{grid_wrapper.grid_state.exchange_runtime.my_rank()}"
            )
            self._tracer.save(f"viztracer{rank}.json")


def tracer(start: int, stop: int) -> tuple[Callable[[], None], Callable[[], None]]:
    ft = _Tracer(start, stop)
    return ft.enter, ft.exit


def init() -> None:
    """Initialize the VizTracer plugin."""

    if "ICON4PY_TRACING_RANGE" in os.environ:
        tracing_range_str = os.environ["ICON4PY_TRACING_RANGE"]
        tracing_range = tracing_range_str.split(":")
        if len(tracing_range) != 2:
            raise ValueError(
                "Invalid format for 'ICON4PY_TRACING_RANGE'. Expected format: 'start:stop'."
            )
        start, stop = int(tracing_range[0]), int(tracing_range[1])
    else:
        # 2 timesteps for a standard setup (with tracing for `solve_nh_run` and `diffusion_run`)
        start, stop = 12, 24

    if "ICON4PY_TRACING_NAMES" in os.environ:
        tracing_functions = os.environ["ICON4PY_TRACING_NAMES"].split(",")
    else:
        tracing_functions = ["solve_nh_run", "diffusion_run"]

    enter, exit_ = tracer(start, stop)

    for name in tracing_functions:
        runtime_config.HOOK_BINDINGS_FUNCTION_ENTER[name] = enter
        runtime_config.HOOK_BINDINGS_FUNCTION_EXIT[name] = exit_
