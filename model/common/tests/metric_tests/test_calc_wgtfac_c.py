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

import pytest

from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate_indices
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.metrics.stencils.calc_wgtfac_c import calc_wgtfac_c
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401  # import fixtures from test_utils package
    data_provider,
    datapath,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    processor_props,
    ranked_data_path,
)
from icon4py.model.common.test_utils.helpers import dallclose, zero_field
from icon4py.model.common.type_alias import wpfloat


@pytest.mark.datatest
def test_calc_wgtfac_c(icon_grid, metrics_savepoint):  # noqa: F811  # fixture
    wgtfac_c = zero_field(icon_grid, CellDim, KDim, dtype=wpfloat)
    wgtfac_c_ref = metrics_savepoint.wgtfac_c()
    z_ifc = metrics_savepoint.z_ifc()
    k_field = _allocate_indices(KDim, grid=icon_grid, is_halfdim=True)

    horizontal_start = 0
    horizontal_end = icon_grid.get_end_index(
        CellDim,
        HorizontalMarkerIndex.end(CellDim),
    )
    vertical_start = 0
    vertical_end = icon_grid.num_levels

    calc_wgtfac_c(
        wgtfac_c,
        z_ifc,
        k_field,
        horizontal_start,
        horizontal_end,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"Koff": KDim},
    )

    assert dallclose(
        wgtfac_c.asnumpy()[:, vertical_start:vertical_end],
        wgtfac_c_ref.asnumpy()[:, vertical_start:vertical_end],
    )
