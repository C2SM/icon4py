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
import importlib
import os
from enum import Enum
from functools import cached_property
from pathlib import Path

import numpy as np
from gt4py.next.program_processors.runners.gtfn import (
    run_gtfn_cached,
    run_gtfn_gpu_cached,
)
from gt4py.next.program_processors.runners.roundtrip import backend as run_roundtrip
from icon4pytools.common.logger import setup_logger


logger = setup_logger(__name__)


class Device(Enum):
    CPU = "CPU"
    GPU = "GPU"


class GT4PyBackend(Enum):
    CPU = "run_gtfn_cached"
    GPU = "run_gtfn_gpu_cached"
    ROUNDTRIP = "run_roundtrip"


def get_local_test_grid():
    test_folder = "testdata"
    module_spec = importlib.util.find_spec("icon4pytools")

    if module_spec and module_spec.origin:
        # following namespace package conventions the root is three levels down
        repo_root = Path(module_spec.origin).parents[3]
        return os.path.join(repo_root, test_folder)
    else:
        raise FileNotFoundError(
            "The `icon4pytools` package could not be found. Ensure the package is installed "
            "and accessible. Alternatively, set the 'ICON_GRID_LOC' environment variable "
            "explicitly to specify the location."
        )


@dataclasses.dataclass
class Icon4PyConfig:
    @cached_property
    def icon4py_backend(self):
        return os.environ.get("ICON4PY_BACKEND", "CPU")

    @cached_property
    def icon_grid_loc(self):
        env_path = os.environ.get("ICON_GRID_LOC")
        if env_path is not None:
            return env_path
        else:
            return get_local_test_grid()

    @cached_property
    def grid_filename(self):
        env_path = os.environ.get("ICON_GRID_NAME")
        if env_path is not None:
            return env_path
        return "grid.nc"

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
        logger.info(f"Using Device = {device}")
        return device
