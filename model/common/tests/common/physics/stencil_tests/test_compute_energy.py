# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.physics.thermodynamics.compute_energy import (
    _compute_dry_static_energy,
    compute_internal_energy_per_area,
    compute_moist_air_heat_capacity_per_area,
)
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import reference_funcs, stencil_tests


class TestComputeInternalEnergyPerArea(stencil_tests.StencilTest):
    PROGRAM = compute_internal_energy_per_area
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        t: np.ndarray,
        qv: np.ndarray,
        qliq: np.ndarray,
        qice: np.ndarray,
        rho: np.ndarray,
        dz: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(out=np.full(t.shape, 38265357.270336017))

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid):
        return dict(
            t=data_alloc.constant_field(255.756, dims.CellDim, dims.KDim, dtype=wpfloat),
            qv=data_alloc.constant_field(0.00122576, dims.CellDim, dims.KDim, dtype=wpfloat),
            qliq=data_alloc.constant_field(1.63837e-20, dims.CellDim, dims.KDim, dtype=wpfloat),
            qice=data_alloc.constant_field(1.09462e-08, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=data_alloc.constant_field(0.83444, dims.CellDim, dims.KDim, dtype=wpfloat),
            dz=data_alloc.constant_field(249.569, dims.CellDim, dims.KDim, dtype=wpfloat),
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KDim: (0, gtx.int32(grid.num_levels)),
            },
            out=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )


class TestComputeDryStaticEnergy(stencil_tests.StencilTest):
    PROGRAM = _compute_dry_static_energy
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        temperature: np.ndarray,
        height_above_ground: np.ndarray,
        grav: float,
        **kwargs,
    ) -> dict:
        dry_static_energy = reference_funcs.compute_dry_static_energy_numpy(
            temperature,
            height_above_ground,
            grav=grav,
        )
        return dict(out=dry_static_energy)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        temperature = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=180.0, high=320.0, dtype=wpfloat
        )
        height_above_ground = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=0.0, high=3.0e4, dtype=wpfloat
        )
        return dict(
            temperature=temperature,
            height_above_ground=height_above_ground,
            grav=constants.GRAV,
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KDim: (0, gtx.int32(grid.num_levels)),
            },
            out=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )


class TestComputeMoistAirHeatCapacityPerArea(stencil_tests.StencilTest):
    PROGRAM = compute_moist_air_heat_capacity_per_area
    OUTPUTS = ("heat_capacity",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        qv: np.ndarray,
        qc: np.ndarray,
        qi: np.ndarray,
        qr: np.ndarray,
        qs: np.ndarray,
        qg: np.ndarray,
        air_mass: np.ndarray,
        **kwargs,
    ) -> dict:
        # get_cvair, mo_aes_phy_diag.f90: the ice heat capacity is the AES one (ci = 2106)
        qliq = qc + qr
        qice = qi + qs + qg
        cv = (
            constants.CVD * (1.0 - (qv + qliq + qice))
            + constants.CVV * qv
            + constants.CPL * qliq
            + constants.SPECIFIC_HEAT_CAPACITY_ICE_PHYSICAL_CONSTANTS * qice
        )
        return dict(heat_capacity=cv * air_mass)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid):
        return dict(
            qv=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=0.0, high=2.0e-2, dtype=wpfloat
            ),
            qc=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=0.0, high=1.0e-3, dtype=wpfloat
            ),
            qi=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=0.0, high=1.0e-3, dtype=wpfloat
            ),
            qr=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=0.0, high=1.0e-3, dtype=wpfloat
            ),
            qs=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=0.0, high=1.0e-3, dtype=wpfloat
            ),
            qg=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=0.0, high=1.0e-3, dtype=wpfloat
            ),
            air_mass=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=1.0, high=1.0e3, dtype=wpfloat
            ),
            heat_capacity=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
