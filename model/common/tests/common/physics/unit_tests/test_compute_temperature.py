# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.grid import simple, vertical as v_grid
from icon4py.model.common.interpolation.stencils import edge_2_cell_vector_rbf_interpolation as rbf
from icon4py.model.common.physics.thermodynamics import (
    compute_pressure,
    compute_temperature,
    compute_tendencies,
)
from icon4py.model.common.states import diagnostic_state as diagnostics, tracer_states as tracers
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, test_utils
from icon4py.model.testing.fixtures.datatest import backend


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


def test_update_exner_and_theta_v(backend: gtx_typing.Backend) -> None:
    """The physics coupling recomputes exner via the exact EOS and theta_v = Tv/exner.
    Tv_new    = Tv + dt * dTv/dt
    exner_new = (rd/p0ref * rho * Tv_new) ** (rd/cpd)
    theta_v   = Tv_new / exner_new
    """
    grid = simple.simple_grid()
    rho_value, tv_value, tv_tendency_value, dtime = 1.1, 280.0, 0.05, 20.0

    rho = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)
    virtual_temperature = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend
    )
    virtual_temperature_tendency = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend
    )
    exner = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)
    theta_v = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=float, allocator=backend)
    rho.ndarray[...] = rho_value
    virtual_temperature.ndarray[...] = tv_value
    virtual_temperature_tendency.ndarray[...] = tv_tendency_value

    compute_temperature.update_exner_and_theta_v.with_backend(backend)(
        rho=rho,
        virtual_temperature=virtual_temperature,
        virtual_temperature_tendency=virtual_temperature_tendency,
        dtime=dtime,
        exner=exner,
        theta_v=theta_v,
        horizontal_start=0,
        horizontal_end=grid.num_cells,
        vertical_start=0,
        vertical_end=grid.num_levels,
        offset_provider={},
    )

    new_virtual_temperature = tv_value + tv_tendency_value * dtime
    expected_exner = np.exp(
        PhysicsConstants.rd_o_cpd
        * np.log(PhysicsConstants.rd_o_p0ref * rho_value * new_virtual_temperature)
    )
    expected_theta_v = new_virtual_temperature / expected_exner

    assert test_utils.dallclose(exner.asnumpy(), expected_exner)
    assert test_utils.dallclose(theta_v.asnumpy(), expected_theta_v)
    # EOS consistency: by definition theta_v * exner == Tv_new
    assert test_utils.dallclose(
        theta_v.asnumpy() * exner.asnumpy(),
        np.full_like(theta_v.asnumpy(), new_virtual_temperature),
    )
