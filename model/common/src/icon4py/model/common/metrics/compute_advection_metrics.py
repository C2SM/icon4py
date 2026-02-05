# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from types import ModuleType

import numpy as np

from icon4py.model.common import field_type_aliases as fa, type_alias as ta


def compute_acvection_deepatmo_fields(
    vct_a: fa.KField[ta.wpfloat],
    nlev: int,
    grid_sphere_radius: float,
    array_ns: ModuleType = np,
):
    deepatmo_divh_mc = array_ns.zeros((nlev,))
    deepatmo_divzU_mc = array_ns.zeros((nlev,))
    deepatmo_divzL_mc = array_ns.zeros((nlev,))
    height_uifc = vct_a[:nlev]
    height_lifc = vct_a[1 : nlev + 1]
    height_mc = 0.5 * (height_lifc + height_uifc)
    radial_distance_mc = height_mc + grid_sphere_radius
    radial_distance_lifc = grid_sphere_radius + height_lifc
    radial_distance_uifc = grid_sphere_radius + height_uifc
    deepatmo_gradh_mc = grid_sphere_radius / radial_distance_mc
    deepatmo_divh_mc = (
        deepatmo_gradh_mc
        * 3.0
        / 4.0
        / (
            1.0
            - radial_distance_lifc
            * radial_distance_uifc
            / (radial_distance_lifc + radial_distance_uifc) ** 2
        )
    )
    deepatmo_divzL_mc = 3.0 / (
        1.0
        + radial_distance_uifc / radial_distance_lifc
        + (radial_distance_uifc / radial_distance_lifc) ** 2
    )
    deepatmo_divzU_mc = 3.0 / (
        1.0
        + radial_distance_lifc / radial_distance_uifc
        + (radial_distance_lifc / radial_distance_uifc) ** 2
    )
    return deepatmo_divh_mc, deepatmo_divzL_mc, deepatmo_divzU_mc
