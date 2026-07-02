# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_energy_from_temperature import (
    compute_energy_from_temperature,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.physics.thermodynamics import ThermodynamicConstants
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def internal_energy_numpy(
    *, t: np.ndarray, qv: np.ndarray, qliq: np.ndarray, qice: np.ndarray
) -> np.ndarray:
    """Reference for 'internal_energy' (mo_aes_thermo.f90) with rho = dz = 1."""
    qtot = qliq + qice + qv
    cv = (
        ThermodynamicConstants.cvd * (1.0 - qtot)
        + ThermodynamicConstants.cvv * qv
        + ThermodynamicConstants.clw * qliq
        + ThermodynamicConstants.ci * qice
    )
    return cv * t - qliq * ThermodynamicConstants.lvc - qice * ThermodynamicConstants.lsc


def energy_from_temperature_reference(
    connectivities: dict[gtx.Dimension, np.ndarray],
    *,
    temperature: np.ndarray,
    qv: np.ndarray,
    qc: np.ndarray,
    qi: np.ndarray,
    qr: np.ndarray,
    qs: np.ndarray,
    qg: np.ndarray,
    height_above_ground: np.ndarray,
    grav: float,
    use_internal_energy: bool,
    **kwargs: Any,
) -> dict:
    if use_internal_energy:
        energy = (
            internal_energy_numpy(t=temperature, qv=qv, qliq=qc + qr, qice=qi + qs + qg)
            + grav * height_above_ground * constants.CVD / constants.CPD
        )
    else:
        energy = constants.CPD * temperature + grav * height_above_ground
    return dict(energy=energy)


def energy_from_temperature_input_data(
    grid: base.Grid, use_internal_energy: bool
) -> dict[str, Any]:
    def moisture_field() -> gtx.Field:
        return data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0e-3, dtype=wpfloat
        )

    return dict(
        temperature=data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=180.0, high=320.0, dtype=wpfloat
        ),
        qv=moisture_field(),
        qc=moisture_field(),
        qi=moisture_field(),
        qr=moisture_field(),
        qs=moisture_field(),
        qg=moisture_field(),
        height_above_ground=data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=3.0e4, dtype=wpfloat
        ),
        energy=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        grav=wpfloat(constants.GRAV),
        use_internal_energy=use_internal_energy,
        horizontal_start=0,
        horizontal_end=gtx.int32(grid.num_cells),
        vertical_start=0,
        vertical_end=gtx.int32(grid.num_levels),
    )


#: Static-params variants: prove that the config bool can be passed both as a regular
#: runtime scalar ("none") and as a static (compile-time) argument selecting the variant.
STATIC_VARIANTS = {
    "none": (),
    "compile_time_variant": ("use_internal_energy",),
}


class TestComputeEnergyFromTemperatureInternal(StencilTest):
    PROGRAM = compute_energy_from_temperature
    OUTPUTS = ("energy",)
    STATIC_PARAMS = STATIC_VARIANTS

    reference = staticmethod(energy_from_temperature_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return energy_from_temperature_input_data(grid, use_internal_energy=True)


class TestComputeEnergyFromTemperatureDryStatic(StencilTest):
    PROGRAM = compute_energy_from_temperature
    OUTPUTS = ("energy",)
    STATIC_PARAMS = STATIC_VARIANTS

    reference = staticmethod(energy_from_temperature_reference)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        return energy_from_temperature_input_data(grid, use_internal_energy=False)
