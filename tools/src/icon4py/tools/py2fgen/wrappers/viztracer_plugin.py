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
- set the environment variable `ICON4PY_TRACING_RANGES` to specify the tracing ranges in the format:
  function_name1:start1:stop1,function_name2:start2:stop2,...
"""

import dataclasses
import os
from collections.abc import Callable

import viztracer  # type: ignore[import-not-found]

from icon4py.tools.py2fgen import runtime_config


@dataclasses.dataclass
class _FunctionTracer:
    start: int
    stop: int
    _tracer: viztracer.VizTracer
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
            # This is writing the file always when a range is finished,
            # but we care only of the last one.
            # The cleaner solution would be to do something on interpreter shutdown,
            # but that would also delay getting the trace in case we only trace the first
            # few steps on a longer run.
            self._tracer.save("viztracer.json")


def function_tracer(
    tracer: viztracer.VizTracer, start: int, stop: int
) -> tuple[Callable[[], None], Callable[[], None]]:
    ft = _FunctionTracer(start, stop, tracer)
    return ft.enter, ft.exit


def init() -> None:
    """Initialize the VizTracer plugin."""

    tracer = viztracer.VizTracer()

    if "ICON4PY_TRACING_RANGES" in os.environ:
        tracing_ranges_str = os.environ["ICON4PY_TRACING_RANGES"]
        tracing_ranges = {
            name: (int(start), int(stop))
            for range_spec in tracing_ranges_str.split(",")
            for name, start, stop in [range_spec.split(":")]
        }
    else:
        # Some useful defaults for the ICON4Py dycore and diffusion granules
        tracing_ranges = {
            "solve_nh_run": (10, 15),
            "diffusion_run": (2, 4),
        }

    for name, (start, stop) in tracing_ranges.items():
        enter, exit_ = function_tracer(tracer, start, stop)
        runtime_config.HOOK_BINDINGS_FUNCTION_ENTER[name] = enter
        runtime_config.HOOK_BINDINGS_FUNCTION_EXIT[name] = exit_
