# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import sqrt, exp

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import geometry, icon as icon_grid
from icon4py.model.common.settings import xp

@gtx.field_operator
def _gaussian_hill(
    cartesian_x: fa.CellField[ta.wpfloat],
    cartesian_y: fa.CellField[ta.wpfloat],
    mount_x: ta.wpfloat,
    mount_y: ta.wpfloat,
    mount_height: ta.wpfloat,
    mount_width: ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:
    """
    Computes the elevation of a parameterized Gaussian hill.
    """
    z_distance = sqrt((cartesian_x - mount_x) ** 2 + (cartesian_y - mount_y) ** 2)
    topography = mount_height * exp(-(z_distance / mount_width) ** 2)
    
    return topography

def gaussian_hill(
    grid: icon_grid.IconGrid,
    cell_geometry: geometry.CellParams,
    mount_lon: ta.wpfloat = -1,
    mount_lat: ta.wpfloat = -1,
    mount_height: ta.wpfloat = 100,
    mount_width: ta.wpfloat = 2000,
) -> fa.CellField[ta.wpfloat]:

    torus_domain_length = 50000
    if mount_lon < 0:
        mount_lon = torus_domain_length / 2
    if mount_lat < 0:
        mount_lat = 0
    topography = gtx.as_field((dims.CellDim,), xp.zeros((grid.num_cells,), dtype=ta.wpfloat))

    _gaussian_hill(
        cartesian_x=cell_geometry.cell_center_lon * torus_domain_length / xp.pi / 2,
        cartesian_y=cell_geometry.cell_center_lat * torus_domain_length / xp.pi / 2,
        mount_x=mount_lon,
        mount_y=mount_lat,
        mount_height=mount_height,
        mount_width=mount_width,
        out=topography,
        offset_provider={},
    )
    return topography