# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.horizontal as h_grid

from . import test_icon


@pytest.mark.parametrize("dim", [dims.C2EDim, dims.C2E2C2EDim, dims.E2VDim, dims.V2EDim, dims.KDim])
def test_domain_raises_for_non_horizontal_dim(dim):
    with pytest.raises(AssertionError) as e:
        h_grid.domain(dim)
    e.match("horizontal dimensions")


def zones():
    for zone in h_grid.Zone.__members__.values():
        yield zone


@pytest.mark.parametrize("dim", test_icon.horizontal_dim())
@pytest.mark.parametrize("zone", zones())
def test_domain_raises_for_invalid_zones(dim, zone):
    print(f"dim={dim}, zone={zone},")
    if dim == dims.CellDim or dim == dims.VertexDim:
        if zone in (
            h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5,
            h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6,
            h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7,
        ):
            with pytest.raises(AssertionError) as e:
                h_grid.domain(dim)(zone)
            e.match("not a valid zone")
