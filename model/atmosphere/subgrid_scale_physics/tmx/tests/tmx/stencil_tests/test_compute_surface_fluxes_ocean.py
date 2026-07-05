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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface.stencils.compute_surface_fluxes_ocean import (
    compute_surface_fluxes_ocean,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.physics.thermodynamics import ThermodynamicConstants as TC
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestComputeSurfaceFluxesOcean(StencilTest):
    PROGRAM = compute_surface_fluxes_ocean
    OUTPUTS = ("evapotranspiration", "latent_hflx", "sensible_hflx", "u_stress", "v_stress")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        rho_sfc: np.ndarray,
        kh: np.ndarray,
        km: np.ndarray,
        wind_rel: np.ndarray,
        qa: np.ndarray,
        qsat_sfc: np.ndarray,
        ta: np.ndarray,
        temperature_sfc: np.ndarray,
        ua: np.ndarray,
        va: np.ndarray,
        ocean_u: np.ndarray,
        ocean_v: np.ndarray,
        wind_gustiness: float,
        **kwargs: Any,
    ) -> dict:
        lvc, cvv, clw, cvd = float(TC.lvc), float(TC.cvv), float(TC.clw), float(TC.cvd)
        gust = np.sqrt(wind_gustiness**2 + wind_rel**2)
        evapotranspiration = rho_sfc * kh * gust * (qa - qsat_sfc)
        latent_hflx = evapotranspiration * (lvc + (cvv - clw) * temperature_sfc)
        sensible_hflx = cvd * rho_sfc * kh * gust * (ta - temperature_sfc)
        u_stress = rho_sfc * km * wind_rel * (ua - ocean_u)
        v_stress = rho_sfc * km * wind_rel * (va - ocean_v)
        return dict(
            evapotranspiration=evapotranspiration,
            latent_hflx=latent_hflx,
            sensible_hflx=sensible_hflx,
            u_stress=u_stress,
            v_stress=v_stress,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        cell = dims.CellDim
        return dict(
            rho_sfc=data_alloc.random_field(grid, cell, low=1.0, high=1.4, dtype=wpfloat),
            kh=data_alloc.random_field(grid, cell, low=1.0e-4, high=0.05, dtype=wpfloat),
            km=data_alloc.random_field(grid, cell, low=1.0e-4, high=0.05, dtype=wpfloat),
            wind_rel=data_alloc.random_field(grid, cell, low=0.3, high=20.0, dtype=wpfloat),
            qa=data_alloc.random_field(grid, cell, low=1.0e-3, high=0.02, dtype=wpfloat),
            qsat_sfc=data_alloc.random_field(grid, cell, low=1.0e-3, high=0.03, dtype=wpfloat),
            ta=data_alloc.random_field(grid, cell, low=240.0, high=310.0, dtype=wpfloat),
            temperature_sfc=data_alloc.random_field(
                grid, cell, low=240.0, high=310.0, dtype=wpfloat
            ),
            ua=data_alloc.random_field(grid, cell, low=-20.0, high=20.0, dtype=wpfloat),
            va=data_alloc.random_field(grid, cell, low=-20.0, high=20.0, dtype=wpfloat),
            ocean_u=data_alloc.random_field(grid, cell, low=-2.0, high=2.0, dtype=wpfloat),
            ocean_v=data_alloc.random_field(grid, cell, low=-2.0, high=2.0, dtype=wpfloat),
            evapotranspiration=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            latent_hflx=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            sensible_hflx=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            u_stress=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            v_stress=data_alloc.zero_field(grid, cell, dtype=wpfloat),
            wind_gustiness=3.0,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )
