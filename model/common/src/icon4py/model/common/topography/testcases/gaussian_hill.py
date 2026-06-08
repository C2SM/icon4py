# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar

from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.grid import grid_manager as gm


@dataclasses.dataclass
class GaussianHillParameters:
    """Parameters for the Gaussian hill test-case topography.

    The default values match TODO (jcanton): add reference to the experiment in ICON.
    """

    mount_x: float = 0.0
    mount_y: float = 0.0
    mount_height: float = 100.0
    mount_width: float = 100.0

    fortran_name_map: ClassVar[dict[str, str]] = {}


def gaussian_hill(
    *,
    parameters: GaussianHillParameters,
    grid_manager: gm.GridManager,
) -> data_alloc.NDArray:
    mount_x = parameters.mount_x
    mount_y = parameters.mount_y
    mount_height = parameters.mount_height
    mount_width = parameters.mount_width

    cell_x = grid_manager.coordinates[dims.CellDim]["x"].ndarray
    cell_y = grid_manager.coordinates[dims.CellDim]["y"].ndarray

    array_ns = data_alloc.array_namespace(cell_x)

    dist = ((cell_x - mount_x) ** 2 + (cell_y - mount_y) ** 2) ** 0.5

    topo = mount_height * array_ns.exp(-((dist / mount_width) ** 2))

    return topo
