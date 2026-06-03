# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from gt4py import next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat

@gtx.field_operator
def _compute_advection_deepatmo_fields(
    height_u: fa.KField[wpfloat],
    height_l: fa.KField[wpfloat],
    grid_sphere_radius: wpfloat,
) -> tuple[fa.KField[wpfloat], fa.KField[wpfloat], fa.KField[wpfloat]]:
    """
    Compute 'deepatmo_divh', 'deepatmo_divzL', 'deepatmo_divzU' from 'vct_a' and 'grid_sphere_radius'.

    # Input Fields:
    - height_u: expects vct_a[:nlev]
    - height_l: expects vct_a[1:nlev+1]

    # Output Fields:
    - deepatmo_divh
    - deepatmo_divzL
    - deepatmo_divzU
    """
    height = wpfloat(0.5) * (height_l + height_u)
    radial_distance = height + grid_sphere_radius
    radial_distance_l = grid_sphere_radius + height_l
    radial_distance_u = grid_sphere_radius + height_u
    deepatmo_gradh = grid_sphere_radius / radial_distance
    deepatmo_divh = (
        deepatmo_gradh
        * wpfloat(3.0)
        / wpfloat(4.0)
        / (
            wpfloat(1.0)
            - radial_distance_l * radial_distance_u / (radial_distance_l + radial_distance_u) ** 2
        )
    )
    deepatmo_divzL = wpfloat(3.0) / (
        wpfloat(1.0) + radial_distance_u / radial_distance_l + (radial_distance_u / radial_distance_l) ** 2
    )
    deepatmo_divzU = wpfloat(3.0) / (
        wpfloat(1.0) + radial_distance_l / radial_distance_u + (radial_distance_l / radial_distance_u) ** 2
    )
    return deepatmo_divh, deepatmo_divzL, deepatmo_divzU


@gtx.program
def compute_advection_deepatmo_fields(
    height_u: fa.KField[wpfloat],
    height_l: fa.KField[wpfloat],
    deepatmo_divh: fa.KField[wpfloat],
    deepatmo_divzL: fa.KField[wpfloat],
    deepatmo_divzU: fa.KField[wpfloat],
    grid_sphere_radius: wpfloat,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Compute 'deepatmo_divh', 'deepatmo_divzL', 'deepatmo_divzU' from 'vct_a' and 'grid_sphere_radius'.

    # Input Fields:
    - height_u: expects vct_a[:nlev]
    - height_l: expects vct_a[1:nlev+1]

    # Inout Fields:
    - deepatmo_divh
    - deepatmo_divzL
    - deepatmo_divzU
    """
    _compute_advection_deepatmo_fields(
        height_u=height_u,
        height_l=height_l,
        grid_sphere_radius=grid_sphere_radius,
        out=(deepatmo_divh, deepatmo_divzL, deepatmo_divzU),
        domain={dims.KDim: (vertical_start, vertical_end)},
    )
