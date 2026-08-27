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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.scalar_diffusion import (
    apply_horizontal_diffusion_and_update_temperature,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def t_from_internal_energy_numpy(
    *, u: np.ndarray, qv: np.ndarray, qliq: np.ndarray, qice: np.ndarray
) -> np.ndarray:
    """Reference for 'T_from_internal_energy' (mo_aes_thermo.f90) with rho = dz = 1."""
    qtot = qliq + qice + qv
    cv = (
        PhysicsConstants.cvd * (1.0 - qtot)
        + PhysicsConstants.cvv * qv
        + PhysicsConstants.cpl * qliq
        + PhysicsConstants.cpi * qice
    )
    return (u + (qliq * PhysicsConstants.lvc + qice * PhysicsConstants.lsc)) / cv


def temperature_from_energy_reference(
    grid: base.Grid,
    *,
    energy: np.ndarray,
    temperature: np.ndarray,
    qv: np.ndarray,
    qc: np.ndarray,
    qi: np.ndarray,
    qr: np.ndarray,
    qs: np.ndarray,
    qg: np.ndarray,
    height_above_ground: np.ndarray,
    grav: float,
    dtime: float,
    use_internal_energy: bool,
    **kwargs: Any,
) -> dict:
    if use_internal_energy:
        u = energy - grav * height_above_ground * constants.CVD / constants.CPD
        new_temperature = t_from_internal_energy_numpy(u=u, qv=qv, qliq=qc + qr, qice=qi + qs + qg)
    else:
        new_temperature = (energy - grav * height_above_ground) / constants.CPD
    tend_temperature = (new_temperature - temperature) * (1.0 / dtime)
    return dict(new_temperature=new_temperature, tend_temperature=tend_temperature)


def horizontal_diffusion_tendency_numpy(
    grid: base.Grid, *, nabla2_flux: np.ndarray, geofac_div: np.ndarray, rho: np.ndarray
) -> np.ndarray:
    """Cell-centered divergence of the horizontal turbulent flux, per unit mass."""
    connectivities = stencil_tests.connectivities_asnumpy(grid)
    c2e = connectivities[dims.C2E]  # (n_cells, 3)
    return np.sum(geofac_div[:, :, np.newaxis] * nabla2_flux[c2e], axis=1) / rho


def on_domain(
    out: np.ndarray,
    computed: np.ndarray,
    *,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
) -> np.ndarray:
    """Value of an output field written by a program only on its domain."""
    result = out.copy()
    result[horizontal_start:horizontal_end, vertical_start:vertical_end] = computed[
        horizontal_start:horizontal_end, vertical_start:vertical_end
    ]
    return result


def apply_horizontal_diffusion_and_update_temperature_reference(
    grid: base.Grid,
    *,
    energy: np.ndarray,
    nabla2_flux: np.ndarray,
    geofac_div: np.ndarray,
    rho: np.ndarray,
    tend_energy: np.ndarray,
    temperature: np.ndarray,
    qv: np.ndarray,
    qc: np.ndarray,
    qi: np.ndarray,
    qr: np.ndarray,
    qs: np.ndarray,
    qg: np.ndarray,
    height_above_ground: np.ndarray,
    new_temperature: np.ndarray,
    tend_temperature: np.ndarray,
    grav: float,
    dtime: float,
    use_internal_energy: bool,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
    **kwargs: Any,
) -> dict:
    domain = dict(
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
    )
    hori_tend = horizontal_diffusion_tendency_numpy(
        grid, nabla2_flux=nabla2_flux, geofac_div=geofac_div, rho=rho
    )
    tend_energy_out = on_domain(tend_energy, tend_energy + hori_tend, **domain)
    # The new energy is internal to the program: it is only fed into the
    # temperature recovery and never written out.
    new_energy = energy + tend_energy_out * dtime
    computed = temperature_from_energy_reference(
        grid,
        energy=new_energy,
        temperature=temperature,
        qv=qv,
        qc=qc,
        qi=qi,
        qr=qr,
        qs=qs,
        qg=qg,
        height_above_ground=height_above_ground,
        grav=grav,
        dtime=dtime,
        use_internal_energy=use_internal_energy,
    )
    return dict(
        tend_energy=tend_energy_out,
        new_temperature=on_domain(new_temperature, computed["new_temperature"], **domain),
        tend_temperature=on_domain(tend_temperature, computed["tend_temperature"], **domain),
    )


def apply_horizontal_diffusion_and_update_temperature_input_data(
    data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid, use_internal_energy: bool
) -> dict[str, Any]:
    # Fortran: tmx 'domain' cell bounds, rl_start = grf_bdywidth_c + 1,
    # rl_end = min_rlcell_int.
    cell_domain = h_grid.domain(dims.CellDim)
    horizontal_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
    horizontal_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
    assert horizontal_start < horizontal_end

    def moisture_field() -> gtx.Field:
        return data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0, high=1.0e-3, dtype=wpfloat)

    return dict(
        energy=data_alloc.random_field(
            dims.CellDim, dims.KDim, low=1.5e5, high=5.0e5, dtype=wpfloat
        ),
        nabla2_flux=data_alloc.random_field(
            dims.EdgeDim, dims.KDim, low=-1.0e3, high=1.0e3, dtype=wpfloat
        ),
        geofac_div=data_alloc.random_field(
            dims.CellDim, dims.C2EDim, low=-1.0e-4, high=1.0e-4, dtype=wpfloat
        ),
        rho=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.5, high=1.4, dtype=wpfloat),
        tend_energy=data_alloc.random_field(
            dims.CellDim, dims.KDim, low=-10.0, high=10.0, dtype=wpfloat
        ),
        temperature=data_alloc.random_field(
            dims.CellDim, dims.KDim, low=180.0, high=320.0, dtype=wpfloat
        ),
        qv=moisture_field(),
        qc=moisture_field(),
        qi=moisture_field(),
        qr=moisture_field(),
        qs=moisture_field(),
        qg=moisture_field(),
        height_above_ground=data_alloc.random_field(
            dims.CellDim, dims.KDim, low=0.0, high=1.0e4, dtype=wpfloat
        ),
        new_temperature=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        tend_temperature=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        grav=wpfloat(constants.GRAV),
        dtime=wpfloat(300.0),
        use_internal_energy=use_internal_energy,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=0,
        vertical_end=gtx.int32(grid.num_levels),
    )


# Static-params variants: prove that the config bool can be passed both as a regular
# runtime scalar ("none") and as a static (compile-time) argument selecting the variant.
STATIC_VARIANTS = {
    "none": (),
    "compile_time_variant": ("use_internal_energy",),
}


class TestApplyHorizontalDiffusionAndUpdateTemperatureInternal(stencil_tests.StencilTest):
    """
    Horizontal energy diffusion fused with the recovery of the temperature from
    the internal energy.

    Outside the computed domain ``tend_energy`` keeps its input values (the
    vertical diffusion tendency) and the two temperature outputs stay zero.
    """

    PROGRAM = apply_horizontal_diffusion_and_update_temperature
    OUTPUTS = ("tend_energy", "new_temperature", "tend_temperature")
    STATIC_PARAMS = STATIC_VARIANTS

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return apply_horizontal_diffusion_and_update_temperature_reference(grid, **kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return apply_horizontal_diffusion_and_update_temperature_input_data(
            data_alloc, grid, use_internal_energy=True
        )


class TestApplyHorizontalDiffusionAndUpdateTemperatureDryStatic(stencil_tests.StencilTest):
    """Same, with the temperature recovered from the dry static energy."""

    PROGRAM = apply_horizontal_diffusion_and_update_temperature
    OUTPUTS = ("tend_energy", "new_temperature", "tend_temperature")
    STATIC_PARAMS = STATIC_VARIANTS

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return apply_horizontal_diffusion_and_update_temperature_reference(grid, **kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        return apply_horizontal_diffusion_and_update_temperature_input_data(
            data_alloc, grid, use_internal_energy=False
        )
