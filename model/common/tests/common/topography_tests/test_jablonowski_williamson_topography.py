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

from icon4py.model.common import dimension as dims
from icon4py.model.common.initialization import jablonowski_williamson_topography as topography
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    import icon4py.model.testing.serialbox as sb


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize("experiment", [definitions.Experiments.JW])
def test_jablonowski_williamson_topography(
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
    grid_savepoint: sb.IconGridSavepoint,
    topography_savepoint: sb.TopographySavepoint,
):
    cell_center_lat = grid_savepoint.lat(dims.CellDim).ndarray
    topo_c = topography.jablonowski_williamson_topography(
        cell_lat=cell_center_lat,
        u0=35.0,
        backend=backend,
    )

    topo_c_ref = topography_savepoint.topo_c().asnumpy()

    assert test_utils.dallclose(
        topo_c,
        topo_c_ref,
    )
