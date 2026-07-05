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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.sea_ice import (
    compute_ice_nonsolar_forcing,
    set_ice_albedo,
    set_ice_temp_zerolayer,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.constants import SeaIceConstants as SI
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


_TMELT = float(constants.MELTING_TEMPERATURE)


class TestComputeIceNonsolarForcing(StencilTest):
    PROGRAM = compute_ice_nonsolar_forcing
    OUTPUTS = ("nonsolar", "dnonsolardt")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        lwflx_net: np.ndarray,
        lhflx: np.ndarray,
        shflx: np.ndarray,
        tsurf_old: np.ndarray,
        emissivity: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        return dict(
            nonsolar=lwflx_net + lhflx + shflx,
            dnonsolardt=-4.0 * emissivity * float(SI.stbo) * tsurf_old**3,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        cell = dims.CellDim
        return dict(
            lwflx_net=data_alloc.random_field(grid, cell, low=-100.0, high=50.0, dtype=wpfloat),
            lhflx=data_alloc.random_field(grid, cell, low=-100.0, high=50.0, dtype=wpfloat),
            shflx=data_alloc.random_field(grid, cell, low=-100.0, high=100.0, dtype=wpfloat),
            tsurf_old=data_alloc.random_field(grid, cell, low=240.0, high=273.0, dtype=wpfloat),
            emissivity=data_alloc.random_field(grid, cell, low=0.9, high=1.0, dtype=wpfloat),
            nonsolar=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            dnonsolardt=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )


class TestSetIceTempZerolayer(StencilTest):
    PROGRAM = set_ice_temp_zerolayer
    OUTPUTS = ("tsurf_new", "qtop", "qbot")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        tsurf_old: np.ndarray,
        hi: np.ndarray,
        hs: np.ndarray,
        swnet: np.ndarray,
        nonsolar: np.ndarray,
        dnonsolardt: np.ndarray,
        freezing_temperature: float,
        heat_capacity_thickness: float,
        nonsolar_gradient_flag: float,
        dtime: float,
        **kwargs: Any,
    ) -> dict:
        ki, ks, rhoi, ci = float(SI.ki), float(SI.ks), float(SI.rhoi), float(SI.ci)
        tsurf_degc = tsurf_old - _TMELT
        c_icelayer = rhoi * heat_capacity_thickness * ci / dtime
        k_eff = ki * ks / np.maximum(ks * hi + ki * hs, 1.0e-12)
        f_s = -k_eff * (tsurf_degc - freezing_temperature)
        f_a = -swnet - nonsolar
        denom = k_eff - nonsolar_gradient_flag * dnonsolardt + c_icelayer
        dtemp_raw = (f_s - f_a) / denom
        is_melting = tsurf_degc + dtemp_raw > 0.0
        dtemp = np.where(is_melting, -tsurf_degc, dtemp_raw)
        has_ice = hi > 0.0
        return dict(
            tsurf_new=np.where(has_ice, tsurf_degc + dtemp + _TMELT, freezing_temperature + _TMELT),
            qtop=np.where(has_ice & is_melting, -f_a + f_s - denom * dtemp, 0.0),
            qbot=np.where(has_ice, -f_s + k_eff * dtemp, 0.0),
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        cell = dims.CellDim
        return dict(
            tsurf_old=data_alloc.random_field(grid, cell, low=245.0, high=273.0, dtype=wpfloat),
            hi=data_alloc.random_field(grid, cell, low=0.1, high=3.0, dtype=wpfloat),
            hs=data_alloc.random_field(grid, cell, low=0.0, high=0.5, dtype=wpfloat),
            swnet=data_alloc.random_field(grid, cell, low=0.0, high=300.0, dtype=wpfloat),
            nonsolar=data_alloc.random_field(grid, cell, low=-150.0, high=100.0, dtype=wpfloat),
            dnonsolardt=data_alloc.random_field(grid, cell, low=-6.0, high=0.0, dtype=wpfloat),
            tsurf_new=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            qtop=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            qbot=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            freezing_temperature=-1.80,
            heat_capacity_thickness=0.10,
            nonsolar_gradient_flag=0.0,
            dtime=1800.0,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )


class TestSetIceAlbedo(StencilTest):
    PROGRAM = set_ice_albedo
    OUTPUTS = ("albvisdir", "albvisdif", "albnirdir", "albnirdif")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        tsurf_new: np.ndarray,
        hi: np.ndarray,
        hs: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        albtrans, albs, albsm, albi, albim = 0.5, 0.85, 0.70, 0.75, 0.70
        tsurf_degc = tsurf_new - _TMELT
        albflag = 1.0 / (1.0 + albtrans * tsurf_degc**2)
        snow = albflag * albsm + (1.0 - albflag) * albs
        ice = albflag * albim + (1.0 - albflag) * albi
        albedo = np.where(hi > 0.0, np.where(hs > 0.01, snow, ice), 0.0)
        return dict(albvisdir=albedo, albvisdif=albedo, albnirdir=albedo, albnirdif=albedo)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        cell = dims.CellDim
        return dict(
            tsurf_new=data_alloc.random_field(grid, cell, low=245.0, high=273.15, dtype=wpfloat),
            hi=data_alloc.random_field(grid, cell, low=0.1, high=3.0, dtype=wpfloat),
            hs=data_alloc.random_field(grid, cell, low=0.0, high=0.5, dtype=wpfloat),
            albvisdir=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            albvisdif=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            albnirdir=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            albnirdif=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )
