# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
    # DaCe Backends
    DACE_CPU = "run_dace_cpu"
    DACE_GPU = "run_dace_gpu"
    DACE_CPU_NOOPT = "run_dace_cpu_noopt"
    DACE_GPU_NOOPT = "run_dace_gpu_noopt"
    # DaCe Orchestration
    DACE_CPU_ORCH = "run_dace_cpu_orch"
    DACE_GPU_ORCH = "run_dace_gpu_orch"
    DACE_CPU_NOOPT_ORCH = "run_dace_cpu_noopt_orch"
    DACE_GPU_NOOPT_ORCH = "run_dace_gpu_noopt_orch"


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
            dace_backend_map = {
                GT4PyBackend.DACE_CPU.name: run_dace_cpu,
                GT4PyBackend.DACE_GPU.name: run_dace_gpu,
                GT4PyBackend.DACE_CPU_NOOPT.name: run_dace_cpu_noopt,
                GT4PyBackend.DACE_GPU_NOOPT.name: run_dace_gpu_noopt,
                # DaCe Orchestration
                GT4PyBackend.DACE_CPU_ORCH.name: run_dace_cpu,
                GT4PyBackend.DACE_GPU_ORCH.name: run_dace_gpu,
                GT4PyBackend.DACE_CPU_NOOPT_ORCH.name: run_dace_cpu_noopt,
                GT4PyBackend.DACE_GPU_NOOPT_ORCH.name: run_dace_gpu_noopt,
            }
            backend_map.update(dace_backend_map)
        return backend_map[self.icon4py_backend]

    @cached_property
    def device(self):
        device_map = {
            GT4PyBackend.CPU.name: Device.CPU,
            GT4PyBackend.GPU.name: Device.GPU,
            GT4PyBackend.ROUNDTRIP.name: Device.CPU,
        }
        if dace:
            dace_device_map = {
                GT4PyBackend.DACE_CPU.name: Device.CPU,
                GT4PyBackend.DACE_GPU.name: Device.GPU,
                GT4PyBackend.DACE_CPU_NOOPT.name: Device.CPU,
                GT4PyBackend.DACE_GPU_NOOPT.name: Device.GPU,
                # DaCe Orchestration
                GT4PyBackend.DACE_CPU_ORCH.name: Device.CPU,
                GT4PyBackend.DACE_GPU_ORCH.name: Device.GPU,
                GT4PyBackend.DACE_CPU_NOOPT_ORCH.name: Device.CPU,
                GT4PyBackend.DACE_GPU_NOOPT_ORCH.name: Device.GPU,
            }
            device_map.update(dace_device_map)
        device = device_map[self.icon4py_backend]
        return device

    @cached_property
    def limited_area(self):
        return os.environ.get("ICON4PY_LAM", False)
