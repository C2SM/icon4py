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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.diagnostics import (
    compute_thermodynamic_diagnostics,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def compute_static_energy_numpy(
    temperature: np.ndarray, height_above_ground: np.ndarray, *, grav: float
) -> np.ndarray:
    return PhysicsConstants.cpd * temperature + grav * height_above_ground


def compute_virtual_potential_temperature_numpy(
    virtual_temperature: np.ndarray, pressure: np.ndarray
) -> np.ndarray:
    return virtual_temperature * (PhysicsConstants.p0ref / pressure) ** PhysicsConstants.rd_o_cpd


def interpolate_cell_field_to_half_levels_with_boundaries_numpy(
    interpolant: np.ndarray,
    wgtfac_c: np.ndarray,
    *,
    wgtfacq1_c_1: np.ndarray,
    wgtfacq1_c_2: np.ndarray,
    wgtfacq1_c_3: np.ndarray,
    wgtfacq_c_1: np.ndarray,
    wgtfacq_c_2: np.ndarray,
    wgtfacq_c_3: np.ndarray,
) -> np.ndarray:
    nlev = interpolant.shape[1]
    interpolation = np.zeros((interpolant.shape[0], nlev + 1), dtype=interpolant.dtype)
    # Fortran jk = 1 (1-based) -> k = 0
    interpolation[:, 0] = (
        wgtfacq1_c_1 * interpolant[:, 0]
        + wgtfacq1_c_2 * interpolant[:, 1]
        + wgtfacq1_c_3 * interpolant[:, 2]
    )
    # Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1
    interpolation[:, 1:nlev] = (
        wgtfac_c[:, 1:nlev] * interpolant[:, 1:nlev]
        + (1.0 - wgtfac_c[:, 1:nlev]) * interpolant[:, 0 : nlev - 1]
    )
    # Fortran jk = nlevp1 (1-based) -> k = nlev
    interpolation[:, nlev] = (
        wgtfacq_c_1 * interpolant[:, nlev - 1]
        + wgtfacq_c_2 * interpolant[:, nlev - 2]
        + wgtfacq_c_3 * interpolant[:, nlev - 3]
    )
    return interpolation


def compute_brunt_vaisala_frequency_numpy(
    theta_v: np.ndarray, wgtfac_c: np.ndarray, inv_ddqz_z_half: np.ndarray, *, grav: float
) -> np.ndarray:
    """Interior half levels k = 1..nlev-1 only; the boundary rows stay zero."""
    nlev = theta_v.shape[1]
    theta_v_ic = (
        wgtfac_c[:, 1:nlev] * theta_v[:, 1:nlev]
        + (1.0 - wgtfac_c[:, 1:nlev]) * theta_v[:, 0 : nlev - 1]
    )
    bruvais = np.zeros((theta_v.shape[0], nlev + 1), dtype=theta_v.dtype)
    bruvais[:, 1:nlev] = (
        grav
        * (theta_v[:, 0 : nlev - 1] - theta_v[:, 1:nlev])
        * inv_ddqz_z_half[:, 1:nlev]
        / theta_v_ic
    )
    return bruvais


class TestComputeThermodynamicDiagnostics(stencil_tests.StencilTest):
    """
    The four cell diagnostics ``Compute_diagnostics`` runs before the first halo
    exchange, fused into one program with one output domain each.

    The horizontal bounds are deliberately distinct (and none of them spans the
    whole field), so an output written on a neighbour's sub-domain is caught.
    ``bruvais`` reads the ``theta_v`` the operator computes, not the (narrower)
    ``theta_v`` output field, so the reference computes it on the full field.
    """

    PROGRAM = compute_thermodynamic_diagnostics
    OUTPUTS = ("static_energy", "theta_v", "rho_ic", "bruvais")
    # The granule binds the vertical bounds and ``nlev`` at compile time; the
    # variant exercises that path, which is also the one dace can specialize.
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_start_interior",
            "vertical_end",
            "vertical_end_half",
            "nlev",
        ),
    }

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        temperature: np.ndarray,
        virtual_temperature: np.ndarray,
        pressure: np.ndarray,
        rho: np.ndarray,
        height_above_ground: np.ndarray,
        wgtfac_c: np.ndarray,
        inv_ddqz_z_half: np.ndarray,
        wgtfacq1_c_1: np.ndarray,
        wgtfacq1_c_2: np.ndarray,
        wgtfacq1_c_3: np.ndarray,
        wgtfacq_c_1: np.ndarray,
        wgtfacq_c_2: np.ndarray,
        wgtfacq_c_3: np.ndarray,
        static_energy: np.ndarray,
        theta_v: np.ndarray,
        rho_ic: np.ndarray,
        bruvais: np.ndarray,
        grav: float,
        nlev: int,
        cell_start_nudging: int,
        cell_start_lateral_boundary_level_2: int,
        cell_start_lateral_boundary_level_3: int,
        cell_end_local: int,
        cell_end_halo_level_2: int,
        **kwargs: Any,
    ) -> dict:
        static_energy_full = compute_static_energy_numpy(
            temperature, height_above_ground, grav=grav
        )
        theta_v_full = compute_virtual_potential_temperature_numpy(virtual_temperature, pressure)
        rho_ic_full = interpolate_cell_field_to_half_levels_with_boundaries_numpy(
            rho,
            wgtfac_c,
            wgtfacq1_c_1=wgtfacq1_c_1,
            wgtfacq1_c_2=wgtfacq1_c_2,
            wgtfacq1_c_3=wgtfacq1_c_3,
            wgtfacq_c_1=wgtfacq_c_1,
            wgtfacq_c_2=wgtfacq_c_2,
            wgtfacq_c_3=wgtfacq_c_3,
        )
        bruvais_full = compute_brunt_vaisala_frequency_numpy(
            theta_v_full, wgtfac_c, inv_ddqz_z_half, grav=grav
        )

        # Each output keeps its initial value outside its own domain.
        static_energy_out = static_energy.copy()
        static_energy_out[cell_start_nudging:cell_end_local, 0:nlev] = static_energy_full[
            cell_start_nudging:cell_end_local, 0:nlev
        ]
        theta_v_out = theta_v.copy()
        theta_v_out[cell_start_lateral_boundary_level_3:cell_end_local, 0:nlev] = theta_v_full[
            cell_start_lateral_boundary_level_3:cell_end_local, 0:nlev
        ]
        rho_ic_out = rho_ic.copy()
        rho_ic_out[cell_start_lateral_boundary_level_2:cell_end_halo_level_2, 0 : nlev + 1] = (
            rho_ic_full[cell_start_lateral_boundary_level_2:cell_end_halo_level_2, 0 : nlev + 1]
        )
        bruvais_out = bruvais.copy()
        bruvais_out[cell_start_lateral_boundary_level_3:cell_end_local, 1:nlev] = bruvais_full[
            cell_start_lateral_boundary_level_3:cell_end_local, 1:nlev
        ]

        return dict(
            static_energy=static_energy_out,
            theta_v=theta_v_out,
            rho_ic=rho_ic_out,
            bruvais=bruvais_out,
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        # Distinct, non-trivial bounds per output: the zones of the simple grid
        # all collapse to (0, num_cells), which would hide a mixed-up domain.
        num_cells = grid.num_cells
        cell_start_lateral_boundary_level_2 = 1
        cell_start_lateral_boundary_level_3 = 3
        cell_start_nudging = 5
        cell_end_local = num_cells - 3
        cell_end_halo_level_2 = num_cells - 1
        assert cell_start_nudging < cell_end_local

        return dict(
            temperature=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=250.0, high=300.0, dtype=wpfloat
            ),
            virtual_temperature=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=250.0, high=300.0, dtype=wpfloat
            ),
            pressure=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=8.0e4, high=1.05e5, dtype=wpfloat
            ),
            rho=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.5, high=1.3, dtype=wpfloat),
            height_above_ground=data_alloc.random_field(
                dims.CellDim, dims.KDim, low=10.0, high=2.0e4, dtype=wpfloat
            ),
            wgtfac_c=data_alloc.random_field(
                dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=wpfloat
            ),
            inv_ddqz_z_half=data_alloc.random_field(
                dims.CellDim,
                dims.KDim,
                low=1.0e-3,
                high=1.0e-1,
                extend={dims.KDim: 1},
                dtype=wpfloat,
            ),
            wgtfacq1_c_1=data_alloc.random_field(dims.CellDim, dtype=wpfloat),
            wgtfacq1_c_2=data_alloc.random_field(dims.CellDim, dtype=wpfloat),
            wgtfacq1_c_3=data_alloc.random_field(dims.CellDim, dtype=wpfloat),
            wgtfacq_c_1=data_alloc.random_field(dims.CellDim, dtype=wpfloat),
            wgtfacq_c_2=data_alloc.random_field(dims.CellDim, dtype=wpfloat),
            wgtfacq_c_3=data_alloc.random_field(dims.CellDim, dtype=wpfloat),
            static_energy=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            theta_v=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            rho_ic=data_alloc.zero_field(
                dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=wpfloat
            ),
            bruvais=data_alloc.zero_field(
                dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=wpfloat
            ),
            grav=wpfloat(constants.GRAV),
            nlev=gtx.int32(grid.num_levels),
            vertical_start=gtx.int32(0),
            vertical_start_interior=gtx.int32(1),
            vertical_end=gtx.int32(grid.num_levels),
            vertical_end_half=gtx.int32(grid.num_levels + 1),
            cell_start_nudging=gtx.int32(cell_start_nudging),
            cell_start_lateral_boundary_level_2=gtx.int32(cell_start_lateral_boundary_level_2),
            cell_start_lateral_boundary_level_3=gtx.int32(cell_start_lateral_boundary_level_3),
            cell_end_local=gtx.int32(cell_end_local),
            cell_end_halo_level_2=gtx.int32(cell_end_halo_level_2),
        )
