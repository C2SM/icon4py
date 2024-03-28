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

import numpy as np
from gt4py.next.program_processors.runners.gtfn import (
    run_gtfn_cached,
    run_gtfn_gpu_cached,
)
from gt4py.next.program_processors.runners.roundtrip import backend as run_roundtrip


class Device(Enum):
    CPU = "CPU"
    ROUNDTRIP = "CPU"
    GPU = "GPU"


class GT4PyBackend(Enum):
    CPU = "run_gtfn_cached"
    GPU = "run_gtfn_gpu_cached"
    ROUNDTRIP = "run_roundtrip"


@dataclasses.dataclass
class Icon4PyConfig:
    @property
    def ICON4PY_BACKEND(self):
        return os.environ.get("ICON4PY_BACKEND", "CPU")

    @property
    def ICON_GRID_LOC(self):
        return os.environ.get("ICON_GRID_LOC", "../../../../testdata")

    @property
    def GRID_FILENAME(self):
        return "grid.nc"

    @property
    def DEVICE(self):
        return Device[self.ICON4PY_BACKEND].value

    @property
    def ARRAY_NS(self):
        if self.ICON4PY_BACKEND == GT4PyBackend.GPU.name:
            import cupy as cp  # type: ignore[import-untyped]

            return cp
        else:
            return np

    @property
    def GT4PY_RUNNER(self):
        backend_map = {
            GT4PyBackend.CPU.name: run_gtfn_cached,
            GT4PyBackend.GPU.name: run_gtfn_gpu_cached,
            GT4PyBackend.ROUNDTRIP.name: run_roundtrip,
        }
        return backend_map[self.ICON4PY_BACKEND]
