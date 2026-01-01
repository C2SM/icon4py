# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

from icon4py.model.common.decomposition import definitions as decomp_defs, halo
from icon4py.model.common.grid import grid_manager as gm


def run_grid_manager_for_singlenode(file: pathlib.Path) -> gm.GridManager:
    manager = _grid_manager(file, NUM_LEVELS)
    manager(
        keep_skip_values=True,
        run_properties=decomp_defs.SingleNodeProcessProperties(),
        decomposer=halo.SingleNodeDecomposer(),
        allocator=None,
    )
    return manager


def _grid_manager(file: pathlib.Path, num_levels: int) -> gm.GridManager:
    manager = gm.GridManager(str(file), num_levels=num_levels)
    return manager


def run_gridmananger_for_multinode(
    file: pathlib.Path,
    run_properties: decomp_defs.ProcessProperties,
    decomposer: halo.Decomposer,
) -> gm.GridManager:
    manager = _grid_manager(file, num_levels=NUM_LEVELS)
    manager(
        keep_skip_values=True, allocator=None, run_properties=run_properties, decomposer=decomposer
    )
    return manager


NUM_LEVELS = 10
