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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_exchange_coefficients import (
    compute_surface_exchange_first_guess,
    obukhov_businger_step,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


_EPS = 1.0e-12
_CKAP = 0.4


def _stability_function(
    richardson: np.ndarray, hz0: np.ndarray, tc: np.ndarray, unstable_coeff: float
) -> np.ndarray:
    stable = 1.0 / (1.0 + 10.0 * richardson * (1.0 + 8.0 * richardson))
    height_factor = (np.maximum(hz0, 1.0) ** (1.0 / 3.0) - 1.0) ** 1.5
    unstable = 1.0 + unstable_coeff * np.abs(richardson) / (
        1.0 + 75.0 * tc * height_factor * np.sqrt(np.abs(richardson))
    )
    return np.where(richardson >= 0.0, stable, unstable)


def _businger_mom(z0: np.ndarray, z1: np.ndarray, length: np.ndarray) -> np.ndarray:
    bsm, bum, half_pi, ln2 = 5.0, 16.0, np.pi / 2.0, np.log(2.0)
    log_ratio = np.log(z1 / z0)
    length_safe = np.where(length == 0.0, 1.0, length)
    zeta, zeta0 = z1 / length_safe, z0 / length_safe
    psi_zeng = -bsm + bsm * zeta0 + (1.0 - bsm) * np.log(np.maximum(zeta, _EPS)) - zeta + 1.0
    stable = np.where(
        zeta > 1.0,
        (log_ratio - psi_zeng) / _CKAP,
        (log_ratio + bsm * zeta - bsm * zeta0) / _CKAP,
    )
    lam = np.sqrt(np.sqrt(np.maximum(1.0 - bum * zeta, _EPS)))
    lam0 = np.sqrt(np.sqrt(np.maximum(1.0 - bum * zeta0, _EPS)))
    psi = (
        2.0 * np.log(1.0 + lam)
        + np.log(1.0 + lam * lam)
        - 2.0 * np.arctan(lam)
        + half_pi
        - 3.0 * ln2
    )
    psi0 = (
        2.0 * np.log(1.0 + lam0)
        + np.log(1.0 + lam0 * lam0)
        - 2.0 * np.arctan(lam0)
        + half_pi
        - 3.0 * ln2
    )
    unstable = (log_ratio - psi + psi0) / _CKAP
    neutral = log_ratio / _CKAP
    return np.where(length > 0.0, stable, np.where(length < 0.0, unstable, neutral))


def _businger_heat(z0: np.ndarray, z1: np.ndarray, length: np.ndarray) -> np.ndarray:
    bsh, buh, ln2 = 5.0, 16.0, np.log(2.0)
    log_ratio = np.log(z1 / z0)
    length_safe = np.where(length == 0.0, 1.0, length)
    zeta, zeta0 = z1 / length_safe, z0 / length_safe
    psi_zeng = -bsh + bsh * zeta0 + (1.0 - bsh) * np.log(np.maximum(zeta, _EPS)) - zeta + 1.0
    stable = np.where(
        zeta > 1.0,
        (log_ratio - psi_zeng) / _CKAP,
        (log_ratio + bsh * zeta - bsh * zeta0) / _CKAP,
    )
    lam = np.sqrt(np.maximum(1.0 - buh * zeta, _EPS))
    lam0 = np.sqrt(np.maximum(1.0 - buh * zeta0, _EPS))
    psi = 2.0 * (np.log(1.0 + lam) - ln2)
    psi0 = 2.0 * (np.log(1.0 + lam0) - ln2)
    unstable = (log_ratio - psi + psi0) / _CKAP
    neutral = log_ratio / _CKAP
    return np.where(length > 0.0, stable, np.where(length < 0.0, unstable, neutral))


class TestComputeSurfaceExchangeFirstGuess(StencilTest):
    PROGRAM = compute_surface_exchange_first_guess
    OUTPUTS = ("km", "kh")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        theta_atm: np.ndarray,
        theta_sfc: np.ndarray,
        wind_rel: np.ndarray,
        rough_m: np.ndarray,
        dz: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_mc = 0.5 * dz
        height_ratio = z_mc / rough_m
        richardson = (
            constants.GRAV * (theta_atm - theta_sfc) * (z_mc - rough_m) / (theta_sfc * wind_rel**2)
        )
        neutral = (_CKAP / np.log(height_ratio)) ** 2
        km = neutral * _stability_function(richardson, height_ratio, neutral, 10.0)
        kh = neutral * _stability_function(richardson, height_ratio, neutral, 15.0)
        return dict(km=km, kh=kh)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        cell = dims.CellDim
        return dict(
            theta_atm=data_alloc.random_field(grid, cell, low=270.0, high=310.0, dtype=wpfloat),
            theta_sfc=data_alloc.random_field(grid, cell, low=270.0, high=310.0, dtype=wpfloat),
            wind_rel=data_alloc.random_field(grid, cell, low=0.5, high=20.0, dtype=wpfloat),
            rough_m=data_alloc.random_field(grid, cell, low=1.0e-4, high=1.0e-2, dtype=wpfloat),
            dz=data_alloc.random_field(grid, cell, low=10.0, high=100.0, dtype=wpfloat),
            km=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            kh=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )


class TestObukhovBusingerStep(StencilTest):
    PROGRAM = obukhov_businger_step
    OUTPUTS = ("km_out", "kh_out")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        km_in: np.ndarray,
        kh_in: np.ndarray,
        theta_atm: np.ndarray,
        theta_sfc: np.ndarray,
        qsat_sfc: np.ndarray,
        qa: np.ndarray,
        wind_rel: np.ndarray,
        rough_m: np.ndarray,
        dz: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_mc = 0.5 * dz
        vtmpc1 = constants.RV_O_RD_MINUS_1
        sensible = kh_in * wind_rel * (theta_sfc - theta_atm)
        latent = kh_in * wind_rel * (qsat_sfc - qa)
        buoyancy_flux = sensible + vtmpc1 * theta_sfc * latent
        ustar = np.sqrt(km_in) * wind_rel
        length = -(ustar**3) * theta_sfc / (constants.GRAV * _CKAP * buoyancy_flux)
        inv_bus_mom = 1.0 / _businger_mom(rough_m, z_mc, length)
        kh_out = inv_bus_mom / _businger_heat(rough_m, z_mc, length)
        km_out = inv_bus_mom * inv_bus_mom
        return dict(km_out=km_out, kh_out=kh_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        cell = dims.CellDim
        return dict(
            km_in=data_alloc.random_field(grid, cell, low=1.0e-3, high=0.1, dtype=wpfloat),
            kh_in=data_alloc.random_field(grid, cell, low=1.0e-3, high=0.1, dtype=wpfloat),
            theta_atm=data_alloc.random_field(grid, cell, low=270.0, high=310.0, dtype=wpfloat),
            theta_sfc=data_alloc.random_field(grid, cell, low=270.0, high=310.0, dtype=wpfloat),
            qsat_sfc=data_alloc.random_field(grid, cell, low=1.0e-3, high=0.03, dtype=wpfloat),
            qa=data_alloc.random_field(grid, cell, low=1.0e-3, high=0.02, dtype=wpfloat),
            wind_rel=data_alloc.random_field(grid, cell, low=0.5, high=20.0, dtype=wpfloat),
            rough_m=data_alloc.random_field(grid, cell, low=1.0e-4, high=1.0e-2, dtype=wpfloat),
            dz=data_alloc.random_field(grid, cell, low=10.0, high=100.0, dtype=wpfloat),
            km_out=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            kh_out=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )
