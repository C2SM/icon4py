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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.energy_update import (
    update_temperature_and_compute_end_of_step_diagnostics,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.constants import PhysicsConstants as phy
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests

from .test_scalar_diffusion import moist_heat_capacity_numpy
from .test_wind_diffusion import on_rows


def internal_energy_per_area_numpy(
    *,
    temperature: np.ndarray,
    qv: np.ndarray,
    q_liquid: np.ndarray,
    q_solid: np.ndarray,
    rho: np.ndarray,
    dz: np.ndarray,
) -> np.ndarray:
    return (
        rho
        * dz
        * (
            moist_heat_capacity_numpy(qv, q_liquid, q_solid) * temperature
            - q_liquid * phy.lvc
            - q_solid * phy.lsc
        )
    )


class TestUpdateTemperatureAndComputeEndOfStepDiagnostics(stencil_tests.StencilTest):
    PROGRAM = update_temperature_and_compute_end_of_step_diagnostics
    OUTPUTS = (
        "dissip_ke",
        "heating",
        "tend_temperature",
        "new_temperature",
        "cptgz",
        "cptgz_vi",
        "dissip_ke_vi",
        "int_energy_vi",
        "int_energy_vi_tend",
        "km",
        "kh",
    )

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        u: np.ndarray,
        v: np.ndarray,
        new_u: np.ndarray,
        new_v: np.ndarray,
        air_mass: np.ndarray,
        cv_air: np.ndarray,
        temperature: np.ndarray,
        tend_temperature: np.ndarray,
        q_snocpymlt: np.ndarray,
        qv: np.ndarray,
        qc: np.ndarray,
        qi: np.ndarray,
        new_qv: np.ndarray,
        new_qc: np.ndarray,
        new_qi: np.ndarray,
        qr: np.ndarray,
        qs: np.ndarray,
        qg: np.ndarray,
        rho: np.ndarray,
        km_ic: np.ndarray,
        kh_ic: np.ndarray,
        ddqz_z_full: np.ndarray,
        height_above_ground: np.ndarray,
        dissipation_factor: float,
        grav: float,
        dtime: float,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        nlev = u.shape[1]
        dissip_ke = (
            0.5 * air_mass * dissipation_factor / dtime * (u**2 - new_u**2 + v**2 - new_v**2)
        )
        heating = dissip_ke.copy()
        heating[:, nlev - 1] -= q_snocpymlt
        new_tend_temperature = tend_temperature + heating / cv_air
        new_temperature = temperature + new_tend_temperature * dtime

        cptgz = phy.cpd * new_temperature + grav * height_above_ground
        new_int_energy = internal_energy_per_area_numpy(
            temperature=new_temperature,
            qv=new_qv,
            q_liquid=new_qc + qr,
            q_solid=new_qi + qs + qg,
            rho=rho,
            dz=ddqz_z_full,
        )
        old_int_energy = internal_energy_per_area_numpy(
            temperature=temperature,
            qv=qv,
            q_liquid=qc + qr,
            q_solid=qi + qs + qg,
            rho=rho,
            dz=ddqz_z_full,
        )
        int_energy_vi = np.cumsum(new_int_energy, axis=1)

        cells = slice(horizontal_start, horizontal_end)
        levels = slice(vertical_start, vertical_end)
        above_surface = slice(vertical_start, vertical_end - 1)
        expected_tend_temperature = tend_temperature.copy()
        expected_tend_temperature[cells, levels] = new_tend_temperature[cells, levels]
        return dict(
            dissip_ke=on_rows(dissip_ke, cells, levels),
            heating=on_rows(heating, cells, levels),
            tend_temperature=expected_tend_temperature,
            new_temperature=on_rows(new_temperature, cells, levels),
            cptgz=on_rows(cptgz, cells, levels),
            cptgz_vi=on_rows(np.cumsum(cptgz * rho * ddqz_z_full, axis=1), cells, levels),
            dissip_ke_vi=on_rows(np.cumsum(dissip_ke, axis=1), cells, levels),
            int_energy_vi=on_rows(int_energy_vi, cells, levels),
            int_energy_vi_tend=on_rows(
                (int_energy_vi - np.cumsum(old_int_energy, axis=1)) / dtime, cells, levels
            ),
            km=on_rows(km_ic[:, 1:], cells, above_surface),
            kh=on_rows(kh_ic[:, 1:], cells, above_surface),
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        def wind() -> gtx.Field:
            return data_alloc.random_field(dims.CellDim, dims.KDim, low=-10.0, high=10.0)

        def mixing_ratio() -> gtx.Field:
            return data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0, high=1.0e-3)

        def temperature() -> gtx.Field:
            return data_alloc.random_field(dims.CellDim, dims.KDim, low=250.0, high=300.0)

        def output() -> gtx.Field:
            return data_alloc.zero_field(dims.CellDim, dims.KDim)

        # narrower than the field, which the grid's zones are not
        horizontal_start, horizontal_end = 1, grid.num_cells - 1
        return dict(
            u=wind(),
            v=wind(),
            new_u=wind(),
            new_v=wind(),
            air_mass=data_alloc.random_field(dims.CellDim, dims.KDim, low=100.0, high=1000.0),
            cv_air=data_alloc.random_field(dims.CellDim, dims.KDim, low=7.0e4, high=7.0e5),
            temperature=temperature(),
            tend_temperature=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=-1.0e-3, high=1.0e-3
            ),
            q_snocpymlt=data_alloc.random_field(dims.CellDim, low=0.0, high=10.0),
            qv=mixing_ratio(),
            qc=mixing_ratio(),
            qi=mixing_ratio(),
            new_qv=mixing_ratio(),
            new_qc=mixing_ratio(),
            new_qi=mixing_ratio(),
            qr=mixing_ratio(),
            qs=mixing_ratio(),
            qg=mixing_ratio(),
            rho=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.5, high=1.3),
            km_ic=data_alloc.random_field(dims.CellDim, dims.KHalfDim, low=0.0, high=10.0),
            kh_ic=data_alloc.random_field(dims.CellDim, dims.KHalfDim, low=0.0, high=10.0),
            ddqz_z_full=data_alloc.random_field(dims.CellDim, dims.KDim, low=100.0, high=1000.0),
            height_above_ground=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=0.0, high=3.0e4
            ),
            dissip_ke=output(),
            heating=output(),
            new_temperature=output(),
            cptgz=output(),
            cptgz_vi=output(),
            dissip_ke_vi=output(),
            int_energy_vi=output(),
            int_energy_vi_tend=output(),
            km=output(),
            kh=output(),
            dissipation_factor=wpfloat(0.8),
            grav=constants.GRAV,
            dtime=wpfloat(300.0),
            nlev=gtx.int32(grid.num_levels),
            horizontal_start=gtx.int32(horizontal_start),
            horizontal_end=gtx.int32(horizontal_end),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )
