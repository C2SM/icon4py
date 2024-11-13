# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# Assuming this code is in a module called icon4py_config.py
import dataclasses
import os
from enum import Enum
from functools import cached_property

import numpy as np
from gt4py.next import itir_python as run_roundtrip
from gt4py.next.program_processors.runners.gtfn import (
    run_gtfn_cached,
    run_gtfn_gpu_cached,
)


try:
    import dace
    from gt4py.next.program_processors.runners.dace import (
        run_dace_cpu,
        run_dace_cpu_noopt,
        run_dace_gpu,
        run_dace_gpu_noopt,
    )
except ImportError:
    from types import ModuleType
    from typing import Optional

    dace: Optional[ModuleType] = None


class Device(Enum):
    CPU = "CPU"
    GPU = "GPU"


class GT4PyBackend(Enum):
    CPU = "run_gtfn_cached"
    GPU = "run_gtfn_gpu_cached"
    ROUNDTRIP = "run_roundtrip"
    DACE_CPU = "run_dace_cpu"
    DACE_GPU = "run_dace_gpu"
    DACE_CPU_NOOPT = "run_dace_cpu_noopt"
    DACE_GPU_NOOPT = "run_dace_gpu_noopt"


@dataclasses.dataclass
class Icon4PyConfig:
    @cached_property
    def icon4py_backend(self):
        backend = os.environ.get("ICON4PY_BACKEND", "CPU")
        if hasattr(GT4PyBackend, backend):
            return backend
        else:
            raise ValueError(
                f"Invalid ICON4Py backend: {backend}. \n"
                f"Available backends: {', '.join([f'{k}' for k in GT4PyBackend.__members__.keys()])}"
            )

    @cached_property
    def icon4py_dace_orchestration(self):
        # Any value other than None will be considered as True
        return os.environ.get("ICON4PY_DACE_ORCHESTRATION", None)

    @cached_property
    def array_ns(self):
        if self.device == Device.GPU:
            import cupy as cp  # type: ignore[import-untyped]

            return cp
        else:
            return np

    @cached_property
    def gt4py_runner(self):
        backend_map = {
            GT4PyBackend.CPU.name: run_gtfn_cached,
            GT4PyBackend.GPU.name: run_gtfn_gpu_cached,
            GT4PyBackend.ROUNDTRIP.name: run_roundtrip,
        }
        if dace:
            backend_map |= {
                GT4PyBackend.DACE_CPU.name: run_dace_cpu,
                GT4PyBackend.DACE_GPU.name: run_dace_gpu,
                GT4PyBackend.DACE_CPU_NOOPT.name: run_dace_cpu_noopt,
                GT4PyBackend.DACE_GPU_NOOPT.name: run_dace_gpu_noopt,
            }
        return backend_map[self.icon4py_backend]

    @cached_property
    def device(self):
        device_map = {
            GT4PyBackend.CPU.name: Device.CPU,
            GT4PyBackend.GPU.name: Device.GPU,
            GT4PyBackend.ROUNDTRIP.name: Device.CPU,
        }
        if dace:
            device_map |= {
                GT4PyBackend.DACE_CPU.name: Device.CPU,
                GT4PyBackend.DACE_GPU.name: Device.GPU,
                GT4PyBackend.DACE_CPU_NOOPT.name: Device.CPU,
                GT4PyBackend.DACE_GPU_NOOPT.name: Device.GPU,
            }
        device = device_map[self.icon4py_backend]
        return device

    @cached_property
    def limited_area(self):
        return os.environ.get("ICON4PY_LAM", False)

    @cached_property
    def parallel_run(self):
        return os.environ.get("ICON4PY_PARALLEL", False)
