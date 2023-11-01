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

import numpy as np
import pytest
from gt4py.next import np_as_located_field

from icon4py.model.common.dimension import EdgeDim, C2EDim, CellDim, CEDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.interpolation.interpolation_fields import (
    compute_c_lin_e,
    compute_geofac_div,
)
from icon4py.model.common.test_utils.helpers import as_1D_sparse_field
from icon4py.model.common.test_utils.datatest_helpers import (  # noqa: F401  # import fixtures from test_utils package
    data_provider,
    datapath,
    download_ser_data,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    processor_props,
    ranked_data_path,
)
from icon4py.model.common.test_utils.helpers import zero_field
from gt4py.next.program_processors.runners.gtfn import run_gtfn

backend = run_gtfn


@pytest.mark.datatest
def test_compute_c_lin_e(
    grid_savepoint, interpolation_savepoint, icon_grid  # noqa: F811  # fixture
):
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    edge_cell_length = grid_savepoint.edge_cell_length()
    owner_mask = grid_savepoint.e_owner_mask()
    c_lin_e_ref = interpolation_savepoint.c_lin_e()
    lateral_boundary = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    c_lin_e = compute_c_lin_e(
        np.asarray(edge_cell_length),
        np.asarray(inv_dual_edge_length),
        np.asarray(owner_mask),
        lateral_boundary,
    )

    assert np.allclose(c_lin_e, c_lin_e_ref)


@pytest.mark.datatest
def test_compute_geofac_div(icon_grid, grid_savepoint, interpolation_savepoint):
    primal_edge_length = 1.0 / grid_savepoint.inverse_primal_edge_lengths()
    edge_orientation = grid_savepoint.edge_orientation()
    orientation = as_1D_sparse_field(edge_orientation, CEDim)
    cell_areas = grid_savepoint.cell_areas()
    geofac_div = zero_field(icon_grid, CellDim, C2EDim)
    geofac_div_new = as_1D_sparse_field(geofac_div, CEDim)
    compute_geofac_div(
        primal_edge_length,
        orientation,
        cell_areas,
        out=geofac_div,
        offset_provider={
            "E2C": icon_grid.get_e2c_connectivity(),
            "C2E": icon_grid.get_c2e_connectivity(),
        },
    )
    # compute_geofac_div.with_backend(backend)(orientation, out=geofac_div, offset_provider={})

    ref = interpolation_savepoint.geofac_div()
    assert np.allclose(geofac_div, orientation)
