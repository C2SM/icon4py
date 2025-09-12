# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from icon4py.model.common.initialization  import jablonowski_williamson_topography as topography
from typing import TYPE_CHECKING
from icon4py.model.testing import test_utils
from icon4py.model.common import dimension as dims
from ..fixtures import *
from icon4py.model.testing import definitions
if TYPE_CHECKING:
    import icon4py.model.testing.serialbox as sb
    import gt4py.next.typing as gtx_typing


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [definitions.Experiments.JW])
def test_jablonowski_williamson_topography(
    experiment,
    backend,
    grid_savepoint,
    topography_savepoint,
):
    cell_center_lat = grid_savepoint.lat(dims.CellDim).asnumpy()
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
