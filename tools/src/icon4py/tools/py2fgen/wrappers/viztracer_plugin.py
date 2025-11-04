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
- set the environment variable PY2FGEN_EXTRA_MODULES=icon4py.tools.py2fgen.wrappers.viztracer_plugin
- set the environment variable ICON4PY_TRACING_RANGES to specify the tracing ranges in the format:
  function_name1:start1:stop1,function_name2:start2:stop2,...
"""

import collections
import dataclasses
import os

import viztracer  # type: ignore[import-not-found]

from icon4py.tools.py2fgen import runtime_config


@dataclasses.dataclass
class Tracer:
    ranges: dict[str, tuple[int, int]] = dataclasses.field(default_factory=dict)
    _counter: dict[str, int] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(int)
    )
    _viztracer: viztracer.VizTracer = dataclasses.field(default_factory=viztracer.VizTracer)

    def enable(self, name: str) -> None:
        if name in self.ranges:
            start, stop = self.ranges[name]
            if start <= self._counter[name] < stop:
                self._viztracer.start()
            self._counter[name] += 1

    def disable(self, name: str) -> None:
        if name in self.ranges:
            start, stop = self.ranges[name]
            if start < self._counter[name] <= stop:
                # "flush_as_finish" to avoid incomplete calls to `disable` which would visualize as nested calls
                self._viztracer.stop(stop_option="flush_as_finish")
            if self._counter[name] == stop:
                # This is writing the file always when a range is finished,
                # but we care only of the last one.
                # The cleaner solution would be to do something on interpreter shutdown,
                # but that would also delay getting the trace in case we only trace the first
                # few steps on a longer run.
                self._viztracer.save("viztracer.json")


tracer = Tracer()
if "ICON4PY_TRACING_RANGES" in os.environ:
    tracing_ranges = os.environ["ICON4PY_TRACING_RANGES"]
    for range_spec in tracing_ranges.split(","):
        name, start_str, stop_str = range_spec.split(":")
        tracer.ranges[name] = (int(start_str), int(stop_str))
else:
    # Some useful defaults for the ICON4Py dycore and diffusion granules
    tracer.ranges["solve_nh_run"] = (10, 15)
    tracer.ranges["diffusion_run"] = (2, 4)

runtime_config.HOOK_BINDINGS_FUNCTION_ENTER = tracer.enable
runtime_config.HOOK_BINDINGS_FUNCTION_EXIT = tracer.disable
