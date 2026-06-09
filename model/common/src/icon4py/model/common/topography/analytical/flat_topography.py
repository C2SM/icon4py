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
class FlatTopographyParameters:
    """Empty parameters class, used for match/case statement in topography.config
    """
    fortran_name_map: ClassVar[dict[str, str]] = {}


def flat_topography(
    *,
    parameters: FlatTopographyParameters,
    grid_manager: gm.GridManager,
) -> data_alloc.NDArray:

    cell_x = grid_manager.coordinates[dims.CellDim]["x"].ndarray

    array_ns = data_alloc.array_namespace(cell_x)

    topo = array_ns.zeros_like(cell_x)

    return topo
