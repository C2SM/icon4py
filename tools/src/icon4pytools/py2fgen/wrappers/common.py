# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
# type: ignore

import dataclasses
import logging
import os
from enum import Enum
from functools import cached_property

import numpy as np
from gt4py.next import itir_python as run_roundtrip
from gt4py.next.program_processors.runners.gtfn import (
    run_gtfn_cached,
    run_gtfn_gpu_cached,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal, icon
from icon4py.model.common.settings import xp


log = logging.getLogger(__name__)

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


config = Icon4PyConfig()
backend = config.gt4py_runner
dace_orchestration = config.icon4py_dace_orchestration
device = config.device
limited_area = config.limited_area


def adjust_fortran_indices(inp: xp.ndarray, offset: int) -> xp.ndarray:
    """For some Fortran arrays we need to subtract 1 to be compatible with Python indexing."""
    return xp.subtract(inp.ndarray, offset)


def construct_icon_grid(
    grid_id,
    global_grid_params,
    num_vertices,
    num_cells,
    num_edges,
    vertical_size,
    limited_area,
    on_gpu,
    cell_starts,
    cell_ends,
    vertex_starts,
    vertex_ends,
    edge_starts,
    edge_ends,
    c2e,
    e2c,
    c2e2c,
    e2c2e,
    e2v,
    v2e,
    v2c,
    e2c2v,
    c2v,
):
    log.debug("Constructing ICON Grid in Python...")
    log.debug("num_cells:%s", num_cells)
    log.debug("num_edges:%s", num_edges)
    log.debug("num_vertices:%s", num_vertices)
    log.debug("num_levels:%s", vertical_size)

    log.debug("Offsetting Fortran connectivitity arrays by 1")
    offset = 1

    cells_start_index = adjust_fortran_indices(cell_starts, offset)
    vertex_start_index = adjust_fortran_indices(vertex_starts, offset)
    edge_start_index = adjust_fortran_indices(edge_starts, offset)

    cells_end_index = cell_ends.ndarray
    vertex_end_index = vertex_ends.ndarray
    edge_end_index = edge_ends.ndarray

    c2e = adjust_fortran_indices(c2e, offset)
    c2v = adjust_fortran_indices(c2v, offset)
    v2c = adjust_fortran_indices(v2c, offset)
    e2v = adjust_fortran_indices(e2v, offset)[
        :, 0:2
    ]  # slicing required for e2v as input data is actually e2c2v
    c2e2c = adjust_fortran_indices(c2e2c, offset)
    v2e = adjust_fortran_indices(v2e, offset)
    e2c2v = adjust_fortran_indices(e2c2v, offset)
    e2c = adjust_fortran_indices(e2c, offset)
    e2c2e = adjust_fortran_indices(e2c2e, offset)

    # stacked arrays
    c2e2c0 = xp.column_stack((xp.asarray(range(c2e2c.shape[0])), c2e2c))
    e2c2e0 = xp.column_stack((xp.asarray(range(e2c2e.shape[0])), e2c2e))

    config = base.GridConfig(
        horizontal_config=horizontal.HorizontalGridSize(
            num_vertices=num_vertices,
            num_cells=num_cells,
            num_edges=num_edges,
        ),
        vertical_size=vertical_size,
        limited_area=limited_area,
        on_gpu=on_gpu,
    )

    grid = (
        icon.IconGrid(id_=grid_id)
        .with_config(config)
        .with_global_params(global_grid_params)
        .with_start_end_indices(dims.VertexDim, vertex_start_index, vertex_end_index)
        .with_start_end_indices(dims.EdgeDim, edge_start_index, edge_end_index)
        .with_start_end_indices(dims.CellDim, cells_start_index, cells_end_index)
        .with_connectivities(
            {
                dims.C2EDim: c2e,
                dims.C2VDim: c2v,
                dims.E2CDim: e2c,
                dims.E2C2EDim: e2c2e,
                dims.C2E2CDim: c2e2c,
                dims.C2E2CODim: c2e2c0,
                dims.E2C2EODim: e2c2e0,
            }
        )
        .with_connectivities(
            {
                dims.V2EDim: v2e,
                dims.E2VDim: e2v,
                dims.E2C2VDim: e2c2v,
                dims.V2CDim: v2c,
            }
        )
    )

    grid.update_size_connectivities(
        {
            dims.ECVDim: grid.size[dims.EdgeDim] * grid.size[dims.E2C2VDim],
            dims.CEDim: grid.size[dims.CellDim] * grid.size[dims.C2EDim],
            dims.ECDim: grid.size[dims.EdgeDim] * grid.size[dims.E2CDim],
        }
    )

    return grid
