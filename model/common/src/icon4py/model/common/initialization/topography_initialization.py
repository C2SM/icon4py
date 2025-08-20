# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import math

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.common import constants as phy_const


def topography_initialization(cell_lat, u0, backend):
    """Function to initialize topography."""
    xp = data_alloc.import_array_ns(backend)
    sin_lat = xp.sin(cell_lat)
    cos_lat = xp.cos(cell_lat)

    eta_0 = 0.252

    fac1 = u0 * xp.cos((1.0 - eta_0) * (math.pi / 2))**1.5
    fac2 = (-2.0 * (sin_lat**6) * (cos_lat**2 + 1.0 / 3.0) + 1.0 / 6.3) * fac1
    fac3 = (
       (1.6 * (cos_lat**3) * (sin_lat**2 + 2.0 / 3.0) - 0.5 * (math.pi / 2))
        * phy_const.EARTH_RADIUS
        * phy_const.EARTH_ANGULAR_VELOCITY
    )
    topo_c = fac1 * (fac2 + fac3) / phy_const.GRAV

    return topo_c
