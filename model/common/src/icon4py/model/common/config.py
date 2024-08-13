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


class Device(Enum):
    CPU = "CPU"
    GPU = "GPU"


class GT4PyBackend(Enum):
    CPU = "run_gtfn_cached"
    GPU = "run_gtfn_gpu_cached"
    ROUNDTRIP = "run_roundtrip"


@dataclasses.dataclass
class Icon4PyConfig:
    @cached_property
    def icon4py_backend(self):
        return os.environ.get("ICON4PY_BACKEND", "CPU")

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
        return backend_map[self.icon4py_backend]

    @cached_property
    def device(self):
        device_map = {
            GT4PyBackend.CPU.name: Device.CPU,
            GT4PyBackend.GPU.name: Device.GPU,
            GT4PyBackend.ROUNDTRIP.name: Device.CPU,
        }
        device = device_map[self.icon4py_backend]
        return device

    @cached_property
    def limited_area(self):
        return os.environ.get("ICON4PY_LAM", False)
