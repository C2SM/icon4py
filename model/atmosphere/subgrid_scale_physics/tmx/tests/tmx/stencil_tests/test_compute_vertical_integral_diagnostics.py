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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_vertical_integral_diagnostics import (
    compute_vertical_integral_diagnostics,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.physics.thermodynamics import ThermodynamicConstants
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def internal_energy_numpy(
    *,
    t: np.ndarray,
    qv: np.ndarray,
    qliq: np.ndarray,
    qice: np.ndarray,
    rho: np.ndarray,
    dz: np.ndarray,
) -> np.ndarray:
    """Reference for 'internal_energy' (mo_aes_thermo.f90)."""
    qtot = qliq + qice + qv
    cv = (
        (
            ThermodynamicConstants.cvd * (1.0 - qtot)
            + ThermodynamicConstants.cvv * qv
            + ThermodynamicConstants.clw * qliq
            + ThermodynamicConstants.ci * qice
        )
        * rho
        * dz
    )
    return cv * t - rho * dz * (
        qliq * ThermodynamicConstants.lvc + qice * ThermodynamicConstants.lsc
    )


class TestComputeVerticalIntegralDiagnostics(StencilTest):
    PROGRAM = compute_vertical_integral_diagnostics
    OUTPUTS = ("cptgz_vi", "dissip_ke_vi", "int_energy_vi", "int_energy_vi_tend")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        static_energy: np.ndarray,
        dissip_ke: np.ndarray,
        rho: np.ndarray,
        dz: np.ndarray,
        temperature: np.ndarray,
        qv: np.ndarray,
        qc: np.ndarray,
        qi: np.ndarray,
        new_temperature: np.ndarray,
        new_qv: np.ndarray,
        new_qc: np.ndarray,
        new_qi: np.ndarray,
        qr: np.ndarray,
        qs: np.ndarray,
        qg: np.ndarray,
        dtime: float,
        **kwargs: Any,
    ) -> dict:
        int_energy_old = internal_energy_numpy(
            t=temperature, qv=qv, qliq=qc + qr, qice=qi + qs + qg, rho=rho, dz=dz
        )
        int_energy_new = internal_energy_numpy(
            t=new_temperature, qv=new_qv, qliq=new_qc + qr, qice=new_qi + qs + qg, rho=rho, dz=dz
        )
        int_energy_vi = np.cumsum(int_energy_new, axis=1)
        return dict(
            cptgz_vi=np.cumsum(static_energy * rho * dz, axis=1),
            dissip_ke_vi=np.cumsum(dissip_ke, axis=1),
            int_energy_vi=int_energy_vi,
            int_energy_vi_tend=(int_energy_vi - np.cumsum(int_energy_old, axis=1)) / dtime,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        def moisture_field() -> gtx.Field:
            return data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=0.0, high=1.0e-3, dtype=wpfloat
            )

        def temperature_field() -> gtx.Field:
            return data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=250.0, high=300.0, dtype=wpfloat
            )

        def output_field() -> gtx.Field:
            return data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            static_energy=data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=1.5e5, high=5.0e5, dtype=wpfloat
            ),
            dissip_ke=data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=-10.0, high=10.0, dtype=wpfloat
            ),
            rho=data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=0.5, high=1.3, dtype=wpfloat
            ),
            dz=data_alloc.random_field(
                grid, dims.CellDim, dims.KDim, low=100.0, high=1000.0, dtype=wpfloat
            ),
            temperature=temperature_field(),
            qv=moisture_field(),
            qc=moisture_field(),
            qi=moisture_field(),
            new_temperature=temperature_field(),
            new_qv=moisture_field(),
            new_qc=moisture_field(),
            new_qi=moisture_field(),
            qr=moisture_field(),
            qs=moisture_field(),
            qg=moisture_field(),
            cptgz_vi=output_field(),
            dissip_ke_vi=output_field(),
            int_energy_vi=output_field(),
            int_energy_vi_tend=output_field(),
            dtime=wpfloat(300.0),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
