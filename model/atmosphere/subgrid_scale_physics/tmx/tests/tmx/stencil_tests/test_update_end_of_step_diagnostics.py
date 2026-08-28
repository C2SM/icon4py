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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.diagnostics import (
    update_end_of_step_diagnostics,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def compute_dry_static_energy_numpy(
    temperature: np.ndarray,
    height_above_ground: np.ndarray,
    *,
    grav: float,
) -> np.ndarray:
    """Reference for 'compute_static_energy' (mo_vdf_atmo.f90)."""
    return constants.CPD * temperature + grav * height_above_ground


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
            PhysicsConstants.cvd * (1.0 - qtot)
            + PhysicsConstants.cvv * qv
            + PhysicsConstants.cpl * qliq
            + PhysicsConstants.cpi * qice
        )
        * rho
        * dz
    )
    return cv * t - rho * dz * (qliq * PhysicsConstants.lvc + qice * PhysicsConstants.lsc)


def vertical_integral_diagnostics_reference(
    *,
    dry_static_energy: np.ndarray,
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
) -> dict:
    """Reference for the vertical-integral part of 'Update_diagnostics'."""
    int_energy_old = internal_energy_numpy(
        t=temperature, qv=qv, qliq=qc + qr, qice=qi + qs + qg, rho=rho, dz=dz
    )
    int_energy_new = internal_energy_numpy(
        t=new_temperature, qv=new_qv, qliq=new_qc + qr, qice=new_qi + qs + qg, rho=rho, dz=dz
    )
    int_energy_vi = np.cumsum(int_energy_new, axis=1)
    return dict(
        cptgz_vi=np.cumsum(dry_static_energy * rho * dz, axis=1),
        dissip_ke_vi=np.cumsum(dissip_ke, axis=1),
        int_energy_vi=int_energy_vi,
        int_energy_vi_tend=(int_energy_vi - np.cumsum(int_energy_old, axis=1)) / dtime,
    )


def exchange_coefficient_diagnostics_reference(
    *,
    km_ic: np.ndarray,
    kh_ic: np.ndarray,
    km_const: float,
    rturb_prandtl: float,
    use_km_const: bool,
) -> dict:
    """Reference for the km/kh part of 'Update_diagnostics'."""
    km = km_ic[:, 1:-1].copy()
    kh = kh_ic[:, 1:-1].copy()
    km_bottom = (km_const, km_const * rturb_prandtl) if use_km_const else (0.0, 0.0)
    km = np.concatenate((km, np.full((km.shape[0], 1), km_bottom[0])), axis=1)
    kh = np.concatenate((kh, np.full((kh.shape[0], 1), km_bottom[1])), axis=1)
    return dict(km=km, kh=kh)


def end_of_step_diagnostics_reference(
    *,
    new_temperature: np.ndarray,
    height_above_ground: np.ndarray,
    dissip_ke: np.ndarray,
    rho: np.ndarray,
    dz: np.ndarray,
    temperature: np.ndarray,
    qv: np.ndarray,
    qc: np.ndarray,
    qi: np.ndarray,
    new_qv: np.ndarray,
    new_qc: np.ndarray,
    new_qi: np.ndarray,
    qr: np.ndarray,
    qs: np.ndarray,
    qg: np.ndarray,
    km_ic: np.ndarray,
    kh_ic: np.ndarray,
    grav: float,
    dtime: float,
    km_const: float,
    rturb_prandtl: float,
    use_km_const: bool,
    **kwargs: Any,
) -> dict:
    # The dry static energy is recomputed from the updated temperature inside the
    # program, and the vertical integrals consume that value, not an input field.
    dry_static_energy = compute_dry_static_energy_numpy(
        new_temperature, height_above_ground, grav=grav
    )
    integrals = vertical_integral_diagnostics_reference(
        dry_static_energy=dry_static_energy,
        dissip_ke=dissip_ke,
        rho=rho,
        dz=dz,
        temperature=temperature,
        qv=qv,
        qc=qc,
        qi=qi,
        new_temperature=new_temperature,
        new_qv=new_qv,
        new_qc=new_qc,
        new_qi=new_qi,
        qr=qr,
        qs=qs,
        qg=qg,
        dtime=dtime,
    )
    coefficients = exchange_coefficient_diagnostics_reference(
        km_ic=km_ic,
        kh_ic=kh_ic,
        km_const=km_const,
        rturb_prandtl=rturb_prandtl,
        use_km_const=use_km_const,
    )
    return dict(dry_static_energy=dry_static_energy, **integrals, **coefficients)


def end_of_step_diagnostics_input_data(
    data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid, use_km_const: bool
) -> dict[str, Any]:
    def moisture_field() -> gtx.Field:
        return data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0, high=1.0e-3, dtype=wpfloat)

    def temperature_field() -> gtx.Field:
        return data_alloc.random_field(
            dims.CellDim, dims.KDim, low=250.0, high=300.0, dtype=wpfloat
        )

    def half_level_field() -> gtx.Field:
        return data_alloc.random_field(
            dims.CellDim,
            dims.KDim,
            low=1.0e-3,
            high=10.0,
            extend={dims.KDim: 1},
            dtype=wpfloat,
        )

    def output_field() -> gtx.Field:
        return data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat)

    return dict(
        new_temperature=temperature_field(),
        height_above_ground=data_alloc.random_field(
            dims.CellDim, dims.KDim, low=0.0, high=3.0e4, dtype=wpfloat
        ),
        dissip_ke=data_alloc.random_field(
            dims.CellDim, dims.KDim, low=-10.0, high=10.0, dtype=wpfloat
        ),
        rho=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.5, high=1.3, dtype=wpfloat),
        dz=data_alloc.random_field(dims.CellDim, dims.KDim, low=100.0, high=1000.0, dtype=wpfloat),
        temperature=temperature_field(),
        qv=moisture_field(),
        qc=moisture_field(),
        qi=moisture_field(),
        new_qv=moisture_field(),
        new_qc=moisture_field(),
        new_qi=moisture_field(),
        qr=moisture_field(),
        qs=moisture_field(),
        qg=moisture_field(),
        km_ic=half_level_field(),
        kh_ic=half_level_field(),
        dry_static_energy=output_field(),
        cptgz_vi=output_field(),
        dissip_ke_vi=output_field(),
        int_energy_vi=output_field(),
        int_energy_vi_tend=output_field(),
        km=output_field(),
        kh=output_field(),
        grav=constants.GRAV,
        dtime=wpfloat(300.0),
        km_const=wpfloat(1.0),
        rturb_prandtl=wpfloat(3.0),
        use_km_const=use_km_const,
        nlev=gtx.int32(grid.num_levels),
        horizontal_start=0,
        horizontal_end=gtx.int32(grid.num_cells),
        vertical_start=0,
        vertical_end=gtx.int32(grid.num_levels),
    )


# Static-params variants: prove that the config bool can be passed both as a regular
# runtime scalar ("none") and as a static (compile-time) argument selecting the variant.
STATIC_VARIANTS = {
    "none": (),
    "compile_time_variant": ("use_km_const",),
}

OUTPUT_FIELDS = (
    "dry_static_energy",
    "cptgz_vi",
    "dissip_ke_vi",
    "int_energy_vi",
    "int_energy_vi_tend",
    "km",
    "kh",
)


class TestUpdateEndOfStepDiagnostics(stencil_tests.StencilTest):
    PROGRAM = update_end_of_step_diagnostics
    OUTPUTS = OUTPUT_FIELDS
    STATIC_PARAMS = STATIC_VARIANTS

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return end_of_step_diagnostics_reference(**kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return end_of_step_diagnostics_input_data(data_alloc, grid, use_km_const=False)


class TestUpdateEndOfStepDiagnosticsKmConst(stencil_tests.StencilTest):
    PROGRAM = update_end_of_step_diagnostics
    OUTPUTS = OUTPUT_FIELDS
    STATIC_PARAMS = STATIC_VARIANTS

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return end_of_step_diagnostics_reference(**kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return end_of_step_diagnostics_input_data(data_alloc, grid, use_km_const=True)
