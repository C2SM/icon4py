# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from types import ModuleType

import numpy as np

from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


def compute_advection_deepatmo_fields(
    vct_a: data_alloc.NDArray,
    nlev: int,
    grid_sphere_radius: float,
    array_ns: ModuleType = np,
) -> tuple[fa.KField[ta.wpfloat], fa.KField[ta.wpfloat], fa.KField[ta.wpfloat]]:
    deepatmo_divh = array_ns.zeros((nlev,))
    deepatmo_divzU = array_ns.zeros((nlev,))
    deepatmo_divzL = array_ns.zeros((nlev,))
    height_u = vct_a[:nlev]
    height_l = vct_a[1 : nlev + 1]
    height = 0.5 * (height_l + height_u)
    radial_distance = height + grid_sphere_radius
    radial_distance_l = grid_sphere_radius + height_l
    radial_distance_u = grid_sphere_radius + height_u
    deepatmo_gradh = grid_sphere_radius / radial_distance
    deepatmo_divh = (
        deepatmo_gradh
        * 3.0
        / 4.0
        / (
            1.0
            - radial_distance_l * radial_distance_u / (radial_distance_l + radial_distance_u) ** 2
        )
    )
    deepatmo_divzL = 3.0 / (
        1.0 + radial_distance_u / radial_distance_l + (radial_distance_u / radial_distance_l) ** 2
    )
    deepatmo_divzU = 3.0 / (
        1.0 + radial_distance_l / radial_distance_u + (radial_distance_l / radial_distance_u) ** 2
    )
    return deepatmo_divh, deepatmo_divzL, deepatmo_divzU
