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
from gt4py.next.program_processors.runners import roundtrip

from icon4py.model.atmosphere.advection.upwind_hflux_miura_cycl_stencil_03b import (
    upwind_hflux_miura_cycl_stencil_03b,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field, zero_field


def upwind_hflux_miura_cycl_stencil_03b_numpy(
    z_tracer_mflx_1_dsl: np.array,
    z_tracer_mflx_2_dsl: np.array,
    z_tracer_mflx_3_dsl: np.array,
):
    p_out_e = (z_tracer_mflx_1_dsl + z_tracer_mflx_2_dsl + z_tracer_mflx_3_dsl) / float(3)
    return p_out_e


def test_upwind_hflux_miura_cycl_stencil_03b():
    grid = SimpleGrid()
    z_tracer_mflx_1_dsl = random_field(grid, EdgeDim, KDim)
    z_tracer_mflx_2_dsl = random_field(grid, EdgeDim, KDim)
    z_tracer_mflx_3_dsl = random_field(grid, EdgeDim, KDim)
    p_out_e = zero_field(grid, EdgeDim, KDim)

    ref = upwind_hflux_miura_cycl_stencil_03b_numpy(
        np.asarray(z_tracer_mflx_1_dsl),
        np.asarray(z_tracer_mflx_2_dsl),
        np.asarray(z_tracer_mflx_3_dsl),
    )

    upwind_hflux_miura_cycl_stencil_03b.with_backend(roundtrip.backend)(
        z_tracer_mflx_1_dsl,
        z_tracer_mflx_2_dsl,
        z_tracer_mflx_3_dsl,
        p_out_e,
        offset_provider={},
    )
    assert np.allclose(ref, p_out_e)
