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

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.land.jsbach.stencils.soil_temperature import (
    soil_ground_heat_flux,
    soil_temperature_back_substitution,
    soil_temperature_coefficients,
)
from icon4py.model.testing.fixtures.datatest import backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid, grid_manager
from icon4py.model.testing.stencil_tests import StencilTest


class TestSoilTemperatureBackSubstitution(StencilTest):
    """Reconstruct the soil temperature column from the (lagged) Richtmyer-Morton
    coefficients: the back-substitution half of the JSBACH soil energy solve.

    Fortran reference: calc_soil_temperature, mo_sse_process.f90:487-504
        t_soil_sl(:,1)    = t_soil_top
        t_soil_sl(:,k+1)  = t_soil_acoef(:,k) + t_soil_bcoef(:,k) * t_soil_sl(:,k)
    """

    PROGRAM = soil_temperature_back_substitution
    OUTPUTS = ("t_soil_sl",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        t_soil_acoef: np.ndarray,
        t_soil_bcoef: np.ndarray,
        t_soil_top: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        nsoil = t_soil_acoef.shape[1]
        t_soil_sl = np.zeros_like(t_soil_acoef)
        t_soil_sl[:, 0] = t_soil_top
        for k in range(1, nsoil):
            t_soil_sl[:, k] = t_soil_acoef[:, k - 1] + t_soil_bcoef[:, k - 1] * t_soil_sl[:, k - 1]
        return dict(t_soil_sl=t_soil_sl)

    @pytest.fixture
    def input_data(self, grid: base_grid.Grid) -> dict:
        # b in [0, 1) keeps the recurrence stable (a physical R&M b-coefficient is a
        # bounded weight); a is an unconstrained temperature-like offset.
        t_soil_acoef = random_field(grid, dims.CellDim, dims.KDim)
        t_soil_bcoef = random_field(grid, dims.CellDim, dims.KDim)
        t_soil_top = random_field(grid, dims.CellDim)
        t_soil_sl = zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            t_soil_acoef=t_soil_acoef,
            t_soil_bcoef=t_soil_bcoef,
            t_soil_top=t_soil_top,
            t_soil_sl=t_soil_sl,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )


class TestSoilTemperatureCoefficients(StencilTest):
    """Forward elimination: build the next-step Richtmyer-Morton A/B coefficients
    from the current soil temperature column.

    Fortran reference: calc_soil_temperature, mo_sse_process.f90:704-743.
    The bottom layer (index nsoil-2) uses the division form (:722-727); the layers
    above use the reciprocal-multiply form (:735-739). Coefficients of the bottom
    layer index (nsoil-1) are unused.
    """

    PROGRAM = soil_temperature_coefficients
    OUTPUTS = ("t_soil_acoef", "t_soil_bcoef")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        t_soil_sl: np.ndarray,
        vol_heat_cap: np.ndarray,
        heat_cond: np.ndarray,
        dz: np.ndarray,
        zd1: np.ndarray,
        delta_time: float,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        nsoil = t_soil_sl.shape[1]
        heat_cap = dz[np.newaxis, :] * vol_heat_cap
        zdz2 = heat_cap / delta_time
        zdz1 = zd1[np.newaxis, :] * heat_cond
        t_soil_acoef = np.zeros_like(t_soil_sl)
        t_soil_bcoef = np.zeros_like(t_soil_sl)
        # bottom layer (division form)
        denom = zdz2[:, nsoil - 1] + zdz1[:, nsoil - 2]
        t_soil_acoef[:, nsoil - 2] = zdz2[:, nsoil - 1] * t_soil_sl[:, nsoil - 1] / denom
        t_soil_bcoef[:, nsoil - 2] = zdz1[:, nsoil - 2] / denom
        # layers above (reciprocal-multiply form)
        for k in range(nsoil - 3, -1, -1):
            z1 = 1.0 / (
                zdz2[:, k + 1] + zdz1[:, k] + zdz1[:, k + 1] * (1.0 - t_soil_bcoef[:, k + 1])
            )
            t_soil_acoef[:, k] = (
                t_soil_sl[:, k + 1] * zdz2[:, k + 1] + zdz1[:, k + 1] * t_soil_acoef[:, k + 1]
            ) * z1
            t_soil_bcoef[:, k] = zdz1[:, k] * z1
        return dict(t_soil_acoef=t_soil_acoef, t_soil_bcoef=t_soil_bcoef)

    @pytest.fixture
    def input_data(self, grid: base_grid.Grid) -> dict:
        # positive heat capacity / conductivity / spacing keep denominators nonzero.
        t_soil_sl = random_field(grid, dims.CellDim, dims.KDim)
        vol_heat_cap = random_field(grid, dims.CellDim, dims.KDim, low=1.0e5, high=3.0e6)
        heat_cond = random_field(grid, dims.CellDim, dims.KDim, low=0.5, high=3.0)
        dz = random_field(grid, dims.KDim, low=0.05, high=6.0)
        zd1 = random_field(grid, dims.KDim, low=0.1, high=5.0)
        # zd1 of the bottom layer is unused by the scheme; pin it to zero.
        zd1_np = zd1.asnumpy()
        zd1_np[grid.num_levels - 1] = 0.0
        zd1 = gtx.as_field((dims.KDim,), zd1_np)
        t_soil_acoef = zero_field(grid, dims.CellDim, dims.KDim)
        t_soil_bcoef = zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            t_soil_sl=t_soil_sl,
            vol_heat_cap=vol_heat_cap,
            heat_cond=heat_cond,
            dz=dz,
            zd1=zd1,
            delta_time=900.0,
            t_soil_acoef=t_soil_acoef,
            t_soil_bcoef=t_soil_bcoef,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )


class TestSoilGroundHeatFlux(StencilTest):
    """Surface diffusive ground heat flux and ground heat capacity from the top-layer
    R&M coefficients.

    Fortran reference: calc_soil_temperature, mo_sse_process.f90:748-751 (evaluated
    at the surface layer). Computed here per level; the caller restricts to the top
    (ground) level.
    """

    PROGRAM = soil_ground_heat_flux
    OUTPUTS = ("grnd_hflx", "hcap_grnd")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        t_soil_sl: np.ndarray,
        t_soil_acoef: np.ndarray,
        t_soil_bcoef: np.ndarray,
        vol_heat_cap: np.ndarray,
        heat_cond: np.ndarray,
        dz: np.ndarray,
        zd1: np.ndarray,
        delta_time: float,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        zdz1 = zd1[np.newaxis, :] * heat_cond
        zdz2 = dz[np.newaxis, :] * vol_heat_cap / delta_time
        grnd_hflx = zdz1 * (t_soil_acoef + (t_soil_bcoef - 1.0) * t_soil_sl)
        hcap_grnd = zdz2 * delta_time + delta_time * (1.0 - t_soil_bcoef) * zdz1
        return dict(grnd_hflx=grnd_hflx, hcap_grnd=hcap_grnd)

    @pytest.fixture
    def input_data(self, grid: base_grid.Grid) -> dict:
        t_soil_sl = random_field(grid, dims.CellDim, dims.KDim)
        t_soil_acoef = random_field(grid, dims.CellDim, dims.KDim)
        t_soil_bcoef = random_field(grid, dims.CellDim, dims.KDim)
        vol_heat_cap = random_field(grid, dims.CellDim, dims.KDim, low=1.0e5, high=3.0e6)
        heat_cond = random_field(grid, dims.CellDim, dims.KDim, low=0.5, high=3.0)
        dz = random_field(grid, dims.KDim, low=0.05, high=6.0)
        zd1 = random_field(grid, dims.KDim, low=0.1, high=5.0)
        grnd_hflx = zero_field(grid, dims.CellDim, dims.KDim)
        hcap_grnd = zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            t_soil_sl=t_soil_sl,
            t_soil_acoef=t_soil_acoef,
            t_soil_bcoef=t_soil_bcoef,
            vol_heat_cap=vol_heat_cap,
            heat_cond=heat_cond,
            dz=dz,
            zd1=zd1,
            delta_time=900.0,
            grnd_hflx=grnd_hflx,
            hcap_grnd=hcap_grnd,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
