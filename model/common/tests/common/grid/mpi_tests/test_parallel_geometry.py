# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.common.grid import geometry_attributes as attrs
from icon4py.model.testing import definitions as test_defs, grid_utils, parallel_helpers, test_utils

from ...fixtures import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    processor_props,
    ranked_data_path,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_distributed_dual_normal_cell(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: test_defs.Experiment,
    processor_props: decomposition.ProcessProperties,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment)
    dual_normal_cell_x_ref = grid_savepoint.dual_normal_cell_x().asnumpy()
    dual_normal_cell_x = grid_geometry.get(attrs.EDGE_TANGENT_CELL_U)
    assert test_utils.dallclose(dual_normal_cell_x.asnumpy(), dual_normal_cell_x_ref, atol=1e-14)
