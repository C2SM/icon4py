# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.diagnostics import (
    assign_constant_viscosity,
    compute_edge_shear_diagnostics,
    compute_scaling_factor_louis,
    compute_smagorinsky_mixing_length,
    compute_smagorinsky_viscosity,
    compute_strain_rate_diagnostics,
    compute_thermodynamic_diagnostics,
    interpolate_km,
)
from icon4py.model.common import constants, dimension as dims, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


def compute_smagorinsky_mixing_length_numpy(
    dz_ic: np.ndarray,
    geopot_agl_ic: np.ndarray,
    cell_area: np.ndarray,
    *,
    smag_constant: float,
    max_turb_scale: float,
    grav: float,
) -> np.ndarray:
    kappa = 0.4
    z_agl = geopot_agl_ic * (1.0 / grav)
    les_filter = smag_constant * np.minimum(
        max_turb_scale, (dz_ic * cell_area[:, np.newaxis]) ** 0.33333
    )
    return (
        (les_filter * z_agl)
        * (les_filter * z_agl)
        / ((les_filter / kappa) * (les_filter / kappa) + z_agl * z_agl)
    )


class TestInitSmagorinskyMixingLength(stencil_tests.StencilTest):
    PROGRAM = compute_smagorinsky_mixing_length
    OUTPUTS = ("mixing_length_sq",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        dz_ic: np.ndarray,
        geopot_agl_ic: np.ndarray,
        cell_area: np.ndarray,
        smag_constant: float,
        max_turb_scale: float,
        grav: float,
        **kwargs,
    ) -> dict:
        mixing_length_sq = compute_smagorinsky_mixing_length_numpy(
            dz_ic,
            geopot_agl_ic,
            cell_area,
            smag_constant=smag_constant,
            max_turb_scale=max_turb_scale,
            grav=grav,
        )
        return dict(mixing_length_sq=mixing_length_sq)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        dz_ic = data_alloc.random_field(
            dims.CellDim, dims.KHalfDim, low=10.0, high=500.0, dtype=wpfloat
        )
        geopot_agl_ic = data_alloc.random_field(
            dims.CellDim, dims.KHalfDim, low=0.0, high=100000.0, dtype=wpfloat
        )
        cell_area = data_alloc.random_field(dims.CellDim, low=1.0e6, high=1.0e8, dtype=wpfloat)
        mixing_length_sq = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat)

        return dict(
            dz_ic=dz_ic,
            geopot_agl_ic=geopot_agl_ic,
            cell_area=cell_area,
            mixing_length_sq=mixing_length_sq,
            smag_constant=wpfloat(0.23),
            max_turb_scale=wpfloat(300.0),
            grav=constants.GRAV,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels + 1),
        )


class TestInitLouisScalingFactor(stencil_tests.StencilTest):
    PROGRAM = compute_scaling_factor_louis
    OUTPUTS = ("scaling_factor_louis",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        cell_area: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(scaling_factor_louis=97294071.23714285 / cell_area)  # mean_cell_area_r2b8

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        cell_area = data_alloc.random_field(dims.CellDim, low=1.0e6, high=1.0e8, dtype=wpfloat)
        scaling_factor_louis = data_alloc.zero_field(dims.CellDim, dtype=wpfloat)

        return dict(
            cell_area=cell_area,
            scaling_factor_louis=scaling_factor_louis,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
        )


def _coefficient_field(
    data_alloc: stencil_tests.DataAllocationWrapper,
    horizontal_dim: gtx.Dimension,
    size: int,
    k_start: int,
) -> gtx.Field:
    """Three quadratic extrapolation coefficient rows, aligned to the levels they multiply."""
    return gtx.as_field(
        gtx.domain({horizontal_dim: (0, size), dims.KDim: (k_start, k_start + 3)}),
        np.random.default_rng().uniform(size=(size, 3)),
        dtype=wpfloat,
        allocator=data_alloc.allocator,
    )


def compute_dry_static_energy_numpy(
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
    wgtfacq1_c: np.ndarray,
    wgtfacq_c: np.ndarray,
) -> np.ndarray:
    nlev = interpolant.shape[1]
    interpolation = np.zeros((interpolant.shape[0], nlev + 1), dtype=interpolant.dtype)
    # Fortran jk = 1 (1-based) -> k = 0
    interpolation[:, 0] = (
        wgtfacq1_c[:, 0] * interpolant[:, 0]
        + wgtfacq1_c[:, 1] * interpolant[:, 1]
        + wgtfacq1_c[:, 2] * interpolant[:, 2]
    )
    # Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1
    interpolation[:, 1:nlev] = (
        wgtfac_c[:, 1:nlev] * interpolant[:, 1:nlev]
        + (1.0 - wgtfac_c[:, 1:nlev]) * interpolant[:, 0 : nlev - 1]
    )
    # Fortran jk = nlevp1 (1-based) -> k = nlev
    interpolation[:, nlev] = (
        wgtfacq_c[:, 2] * interpolant[:, nlev - 1]
        + wgtfacq_c[:, 1] * interpolant[:, nlev - 2]
        + wgtfacq_c[:, 0] * interpolant[:, nlev - 3]
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
    OUTPUTS = ("dry_static_energy", "theta_v", "rho_ic", "bruvais")
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
        wgtfacq1_c: np.ndarray,
        wgtfacq_c: np.ndarray,
        dry_static_energy: np.ndarray,
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
        dry_static_energy_full = compute_dry_static_energy_numpy(
            temperature, height_above_ground, grav=grav
        )
        theta_v_full = compute_virtual_potential_temperature_numpy(virtual_temperature, pressure)
        rho_ic_full = interpolate_cell_field_to_half_levels_with_boundaries_numpy(
            rho,
            wgtfac_c,
            wgtfacq1_c=wgtfacq1_c,
            wgtfacq_c=wgtfacq_c,
        )
        bruvais_full = compute_brunt_vaisala_frequency_numpy(
            theta_v_full, wgtfac_c, inv_ddqz_z_half, grav=grav
        )

        # Each output keeps its initial value outside its own domain.
        dry_static_energy_out = dry_static_energy.copy()
        dry_static_energy_out[cell_start_nudging:cell_end_local, 0:nlev] = dry_static_energy_full[
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
            dry_static_energy=dry_static_energy_out,
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
            wgtfac_c=data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat),
            inv_ddqz_z_half=data_alloc.random_field(
                dims.CellDim, dims.KHalfDim, low=0.001, high=0.1, dtype=wpfloat
            ),
            wgtfacq1_c=_coefficient_field(data_alloc, dims.CellDim, grid.num_cells, 0),
            wgtfacq_c=_coefficient_field(
                data_alloc, dims.CellDim, grid.num_cells, grid.num_levels - 3
            ),
            dry_static_energy=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            theta_v=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            rho_ic=data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat),
            bruvais=data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat),
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


def cell_2_edge_interpolation_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    in_field: np.ndarray,
    coeff: np.ndarray,
) -> np.ndarray:
    """Reference of ``_cell_2_edge_interpolation`` (w -> w_ie)."""
    e2c = connectivities[dims.E2C]  # (n_edges, 2)
    return np.sum(in_field[e2c] * np.expand_dims(coeff, axis=-1), axis=1)


def interpolate_edge_field_to_half_levels_with_boundaries_numpy(
    *,
    interpolant: np.ndarray,
    wgtfac_e: np.ndarray,
    wgtfacq1_e: np.ndarray,
    wgtfacq_e: np.ndarray,
) -> np.ndarray:
    """Reference of ``_interpolate_edge_field_to_half_levels_with_boundaries_wp`` (vn -> vn_ie)."""
    nlev = interpolant.shape[1]
    interpolation = np.zeros((interpolant.shape[0], nlev + 1), dtype=interpolant.dtype)
    interpolation[:, 0] = (
        wgtfacq1_e[:, 0] * interpolant[:, 0]
        + wgtfacq1_e[:, 1] * interpolant[:, 1]
        + wgtfacq1_e[:, 2] * interpolant[:, 2]
    )
    interpolation[:, 1:nlev] = (
        wgtfac_e[:, 1:nlev] * interpolant[:, 1:nlev]
        + (1.0 - wgtfac_e[:, 1:nlev]) * interpolant[:, 0 : nlev - 1]
    )
    interpolation[:, nlev] = (
        wgtfacq_e[:, 2] * interpolant[:, nlev - 1]
        + wgtfacq_e[:, 1] * interpolant[:, nlev - 2]
        + wgtfacq_e[:, 0] * interpolant[:, nlev - 3]
    )
    return interpolation


def compute_tangential_wind_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    vn: np.ndarray,
    rbf_vec_coeff_e: np.ndarray,
) -> np.ndarray:
    """Reference of ``_compute_tangential_wind_wp`` (vn_ie -> vt_ie)."""
    e2c2e = connectivities[dims.E2C2E]  # (n_edges, 4)
    return np.sum(vn[e2c2e] * np.expand_dims(rbf_vec_coeff_e, axis=-1), axis=1)


def compute_shear_and_div_of_stress_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    *,
    u_vert: np.ndarray,
    v_vert: np.ndarray,
    w_vert: np.ndarray,
    w: np.ndarray,
    vn_ie: np.ndarray,
    vt_ie: np.ndarray,
    w_ie: np.ndarray,
    primal_normal_vert_x: np.ndarray,
    primal_normal_vert_y: np.ndarray,
    dual_normal_vert_x: np.ndarray,
    dual_normal_vert_y: np.ndarray,
    tangent_orientation: np.ndarray,
    inv_primal_edge_length: np.ndarray,
    inv_vert_vert_length: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    inv_ddqz_z_full_e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference of ``_compute_shear_and_div_of_stress`` (verbatim from the pre-fusion test)."""
    e2c2v = connectivities[dims.E2C2V]  # (n_edges, 4)
    e2c = connectivities[dims.E2C]  # (n_edges, 2)

    # (n_edges, 4, nlev) gathers of the vertex velocities
    u_vert_e = u_vert[e2c2v]
    v_vert_e = v_vert[e2c2v]

    # (n_edges, 4, 1) geometrical factors per E2C2V neighbor
    pn_x = np.expand_dims(primal_normal_vert_x, axis=-1)
    pn_y = np.expand_dims(primal_normal_vert_y, axis=-1)
    dn_x = np.expand_dims(dual_normal_vert_x, axis=-1)
    dn_y = np.expand_dims(dual_normal_vert_y, axis=-1)

    # (n_edges, 1) edge geometry
    tang = np.expand_dims(tangent_orientation, axis=-1)
    inv_pel = np.expand_dims(inv_primal_edge_length, axis=-1)
    inv_vvl = np.expand_dims(inv_vert_vert_length, axis=-1)
    inv_del = np.expand_dims(inv_dual_edge_length, axis=-1)

    # Normal/tangential velocity components at the four vertices, (n_edges, 4, nlev)
    vn_vert = u_vert_e * pn_x + v_vert_e * pn_y
    vt_vert = u_vert_e * dn_x + v_vert_e * dn_y

    # Vertical wind at full levels: cells (E2C) and edge endpoints (E2C2V 0, 1)
    w_c = w[e2c]  # (n_edges, 2, nlev + 1)
    w_full_c = 0.5 * (w_c[:, :, :-1] + w_c[:, :, 1:])  # (n_edges, 2, nlev)
    w_v = w_vert[e2c2v[:, 0:2]]  # (n_edges, 2, nlev + 1)
    w_full_v = 0.5 * (w_v[:, :, :-1] + w_v[:, :, 1:])  # (n_edges, 2, nlev)

    # Velocity gradient tensor at edge of full levels
    vgrad_11 = (vn_vert[:, 3] - vn_vert[:, 2]) * inv_vvl
    vgrad_12 = (vn_vert[:, 1] - vn_vert[:, 0]) * tang * inv_pel
    vgrad_13 = (vn_ie[:, :-1] - vn_ie[:, 1:]) * inv_ddqz_z_full_e

    vgrad_21 = (vt_vert[:, 3] - vt_vert[:, 2]) * inv_vvl
    vgrad_22 = (vt_vert[:, 1] - vt_vert[:, 0]) * tang * inv_pel
    vgrad_23 = (vt_ie[:, :-1] - vt_ie[:, 1:]) * inv_ddqz_z_full_e

    vgrad_31 = (w_full_c[:, 1] - w_full_c[:, 0]) * inv_del
    vgrad_32 = (w_full_v[:, 1] - w_full_v[:, 0]) * tang * inv_pel
    vgrad_33 = (w_ie[:, :-1] - w_ie[:, 1:]) * inv_ddqz_z_full_e

    # Strain rates at edge center
    d_12 = vgrad_12 + vgrad_21
    d_13 = vgrad_13 + vgrad_31
    d_23 = vgrad_23 + vgrad_32

    shear = 4.0 * (vgrad_11**2 + vgrad_22**2 + vgrad_33**2) + 2.0 * (d_12**2 + d_13**2 + d_23**2)
    div_stress = vgrad_11 + vgrad_22 + vgrad_33

    return shear, div_stress


def _on_subdomain(
    initial: np.ndarray,
    computed: np.ndarray,
    horizontal: tuple[int, int],
    vertical: tuple[int, int],
) -> np.ndarray:
    """The program's per-output domain: outside it the output keeps its initial value."""
    out = initial.copy()
    horizontal_slice = slice(*horizontal)
    vertical_slice = slice(*vertical)
    out[horizontal_slice, vertical_slice] = computed[horizontal_slice, vertical_slice]
    return out


class TestComputeEdgeShearDiagnostics(stencil_tests.StencilTest):
    PROGRAM = compute_edge_shear_diagnostics
    OUTPUTS = ("w_ie", "vn_ie", "vt_ie", "shear", "div_stress")
    # The granule binds the vertical bounds and ``nlev`` at compile time; the
    # variant exercises that path, which is also the one dace can specialize.
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "vertical_end_half",
            "nlev",
        ),
    }

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        w: np.ndarray,
        vn: np.ndarray,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        w_vert: np.ndarray,
        c_lin_e: np.ndarray,
        wgtfac_e: np.ndarray,
        wgtfacq1_e: np.ndarray,
        wgtfacq_e: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        primal_normal_vert_x: np.ndarray,
        primal_normal_vert_y: np.ndarray,
        dual_normal_vert_x: np.ndarray,
        dual_normal_vert_y: np.ndarray,
        tangent_orientation: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        inv_ddqz_z_full_e: np.ndarray,
        w_ie: np.ndarray,
        vn_ie: np.ndarray,
        vt_ie: np.ndarray,
        shear: np.ndarray,
        div_stress: np.ndarray,
        nlev: int,
        edge_start_lateral_boundary_level_2: int,
        edge_start_lateral_boundary_level_3: int,
        edge_start_lateral_boundary_level_4: int,
        edge_end_halo_level_2: int,
        edge_end_halo_level_3: int,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)

        # The fused field operator evaluates the intermediates wherever a consumer
        # needs them, independently of the sub-domain each of them is written on.
        w_ie_full = cell_2_edge_interpolation_numpy(connectivities, in_field=w, coeff=c_lin_e)
        vn_ie_full = interpolate_edge_field_to_half_levels_with_boundaries_numpy(
            interpolant=vn,
            wgtfac_e=wgtfac_e,
            wgtfacq1_e=wgtfacq1_e,
            wgtfacq_e=wgtfacq_e,
        )
        vt_ie_full = compute_tangential_wind_numpy(
            connectivities, vn=vn_ie_full, rbf_vec_coeff_e=rbf_vec_coeff_e
        )
        shear_full, div_stress_full = compute_shear_and_div_of_stress_numpy(
            connectivities,
            u_vert=u_vert,
            v_vert=v_vert,
            w_vert=w_vert,
            w=w,
            vn_ie=vn_ie_full,
            vt_ie=vt_ie_full,
            w_ie=w_ie_full,
            primal_normal_vert_x=primal_normal_vert_x,
            primal_normal_vert_y=primal_normal_vert_y,
            dual_normal_vert_x=dual_normal_vert_x,
            dual_normal_vert_y=dual_normal_vert_y,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_ddqz_z_full_e=inv_ddqz_z_full_e,
        )

        all_half_levels = (0, nlev + 1)
        all_full_levels = (0, nlev)
        return dict(
            w_ie=_on_subdomain(
                w_ie,
                w_ie_full,
                (edge_start_lateral_boundary_level_2, edge_end_halo_level_2),
                all_half_levels,
            ),
            vn_ie=_on_subdomain(
                vn_ie,
                vn_ie_full,
                (edge_start_lateral_boundary_level_2, edge_end_halo_level_3),
                all_half_levels,
            ),
            vt_ie=_on_subdomain(
                vt_ie,
                vt_ie_full,
                (edge_start_lateral_boundary_level_3, edge_end_halo_level_2),
                all_half_levels,
            ),
            shear=_on_subdomain(
                shear,
                shear_full,
                (edge_start_lateral_boundary_level_4, edge_end_halo_level_2),
                all_full_levels,
            ),
            div_stress=_on_subdomain(
                div_stress,
                div_stress_full,
                (edge_start_lateral_boundary_level_4, edge_end_halo_level_2),
                all_full_levels,
            ),
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        w = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat)
        vn = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        u_vert = data_alloc.random_field(dims.VertexDim, dims.KDim, dtype=ta.wpfloat)
        v_vert = data_alloc.random_field(dims.VertexDim, dims.KDim, dtype=ta.wpfloat)
        w_vert = data_alloc.random_field(dims.VertexDim, dims.KHalfDim, dtype=ta.wpfloat)

        c_lin_e = data_alloc.random_field(dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat)
        wgtfac_e = data_alloc.random_field(dims.EdgeDim, dims.KHalfDim, dtype=ta.wpfloat)
        wgtfacq1_e = _coefficient_field(data_alloc, dims.EdgeDim, grid.num_edges, 0)
        wgtfacq_e = _coefficient_field(
            data_alloc, dims.EdgeDim, grid.num_edges, grid.num_levels - 3
        )
        rbf_vec_coeff_e = data_alloc.random_field(dims.EdgeDim, dims.E2C2EDim, dtype=ta.wpfloat)

        primal_normal_vert_x = data_alloc.random_field(
            dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        primal_normal_vert_y = data_alloc.random_field(
            dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        dual_normal_vert_x = data_alloc.random_field(dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat)
        dual_normal_vert_y = data_alloc.random_field(dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat)

        tangent_orientation = data_alloc.random_sign(dims.EdgeDim, dtype=ta.wpfloat)
        inv_primal_edge_length = data_alloc.random_field(dims.EdgeDim, dtype=ta.wpfloat)
        inv_vert_vert_length = data_alloc.random_field(dims.EdgeDim, dtype=ta.wpfloat)
        inv_dual_edge_length = data_alloc.random_field(dims.EdgeDim, dtype=ta.wpfloat)
        inv_ddqz_z_full_e = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        w_ie = data_alloc.zero_field(dims.EdgeDim, dims.KHalfDim, dtype=ta.wpfloat)
        vn_ie = data_alloc.zero_field(dims.EdgeDim, dims.KHalfDim, dtype=ta.wpfloat)
        vt_ie = data_alloc.zero_field(dims.EdgeDim, dims.KHalfDim, dtype=ta.wpfloat)
        shear = data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        div_stress = data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        # Fortran rl bounds of the fused subroutines (mo_vdf_atmo.f90):
        # cells2edges_scalar (w_ie) 2..min_rledge_int-2,
        # interpolate_normal_velocity_edge_interface (vn_ie) 2..min_rledge_int-3,
        # rbf_vec_interpol_edge (vt_ie) 3..min_rledge_int-2,
        # compute_velocity_gradient_tensor / compute_shear 4..min_rledge_int-2.
        edge_domain = h_grid.domain(dims.EdgeDim)
        edge_start_lateral_boundary_level_2 = grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        edge_start_lateral_boundary_level_3 = grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        edge_start_lateral_boundary_level_4 = grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        edge_end_halo_level_2 = grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
        edge_end_halo_level_3 = grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_3))
        assert edge_start_lateral_boundary_level_4 < edge_end_halo_level_2

        return dict(
            w=w,
            vn=vn,
            u_vert=u_vert,
            v_vert=v_vert,
            w_vert=w_vert,
            c_lin_e=c_lin_e,
            wgtfac_e=wgtfac_e,
            wgtfacq1_e=wgtfacq1_e,
            wgtfacq_e=wgtfacq_e,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            primal_normal_vert_x=primal_normal_vert_x,
            primal_normal_vert_y=primal_normal_vert_y,
            dual_normal_vert_x=dual_normal_vert_x,
            dual_normal_vert_y=dual_normal_vert_y,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_ddqz_z_full_e=inv_ddqz_z_full_e,
            w_ie=w_ie,
            vn_ie=vn_ie,
            vt_ie=vt_ie,
            shear=shear,
            div_stress=div_stress,
            nlev=gtx.int32(grid.num_levels),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
            vertical_end_half=gtx.int32(grid.num_levels + 1),
            edge_start_lateral_boundary_level_2=gtx.int32(edge_start_lateral_boundary_level_2),
            edge_start_lateral_boundary_level_3=gtx.int32(edge_start_lateral_boundary_level_3),
            edge_start_lateral_boundary_level_4=gtx.int32(edge_start_lateral_boundary_level_4),
            edge_end_halo_level_2=gtx.int32(edge_end_halo_level_2),
            edge_end_halo_level_3=gtx.int32(edge_end_halo_level_3),
        )


def interpolate_to_cell_center_numpy(
    interpolant: np.ndarray, e_bln_c_s: np.ndarray, c2e: np.ndarray
) -> np.ndarray:
    """Edge -> cell average with the bilinear C2E weights, on full levels."""
    return np.sum(np.expand_dims(e_bln_c_s, axis=-1) * interpolant[c2e], axis=1)


def interpolate_shear_to_half_level_cells_numpy(
    shear: np.ndarray, e_bln_c_s: np.ndarray, wgtfac_c: np.ndarray, c2e: np.ndarray
) -> np.ndarray:
    """Reference of ``_interpolate_edge_field_to_cell_half_levels_wp`` (nlev + 1 levels)."""
    shear_c = interpolate_to_cell_center_numpy(shear, e_bln_c_s, c2e)

    # Full -> half level interpolation: half level k mixes full levels k and k - 1.
    # Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1 (0-based); the top and
    # bottom half-level rows are not computed.
    mech_prod = np.zeros_like(wgtfac_c)
    mech_prod[:, 1:-1] = (
        wgtfac_c[:, 1:-1] * shear_c[:, 1:] + (1.0 - wgtfac_c[:, 1:-1]) * shear_c[:, :-1]
    )
    return mech_prod


class TestComputeStrainRateDiagnostics(stencil_tests.StencilTest):
    PROGRAM = compute_strain_rate_diagnostics
    OUTPUTS = ("div_c", "mech_prod")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        shear: np.ndarray,
        div_stress: np.ndarray,
        e_bln_c_s: np.ndarray,
        wgtfac_c: np.ndarray,
        div_c: np.ndarray,
        mech_prod: np.ndarray,
        vertical_end: int,
        cell_start_nudging: int,
        cell_start_lateral_boundary_level_3: int,
        cell_end_halo: int,
        **kwargs: Any,
    ) -> dict:
        nlev = vertical_end
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        c2e = connectivities[dims.C2E]  # (n_cells, 3)

        div_c_full = interpolate_to_cell_center_numpy(div_stress, e_bln_c_s, c2e)
        mech_prod_full = interpolate_shear_to_half_level_cells_numpy(
            shear, e_bln_c_s, wgtfac_c, c2e
        )

        # Each output only covers its own sub-domain; elsewhere the field keeps the
        # value it was allocated with.
        div_c_out = div_c.copy()
        div_c_out[cell_start_nudging:cell_end_halo, 0:nlev] = div_c_full[
            cell_start_nudging:cell_end_halo, 0:nlev
        ]

        mech_prod_out = mech_prod.copy()
        mech_prod_out[cell_start_lateral_boundary_level_3:cell_end_halo, 1:nlev] = mech_prod_full[
            cell_start_lateral_boundary_level_3:cell_end_halo, 1:nlev
        ]

        return dict(div_c=div_c_out, mech_prod=mech_prod_out)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, Any]:
        shear = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        div_stress = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        e_bln_c_s = data_alloc.random_field(dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)
        wgtfac_c = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat)
        div_c = data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        mech_prod = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat)

        # Fortran: get_horizontal_divergence_strain_rate_cell (div_c) starts at
        # refin_ctrl grf_bdywidth_c + 1 and interpolate_rate_of_strain_full2half_edge2cell
        # (mech_prod) at refin_ctrl 3; both end at min_rlcell_int - 1.
        cell_domain = h_grid.domain(dims.CellDim)
        cell_start_nudging = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        cell_start_lateral_boundary_level_3 = grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        cell_end_halo = grid.end_index(cell_domain(h_grid.Zone.HALO))
        # A grid without a lateral boundary (the simple grid) starts every cell zone
        # at 0, which would make the two per-output horizontal domains coincide.
        # Pull the div_c start in by one cell so that both outputs are masked with
        # their own bound, as they are on a regional grid where the nudging zone
        # starts well after lateral boundary level 3.
        cell_start_nudging = max(cell_start_nudging, cell_start_lateral_boundary_level_3 + 1)
        assert cell_start_lateral_boundary_level_3 < cell_start_nudging < cell_end_halo

        return dict(
            shear=shear,
            div_stress=div_stress,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            div_c=div_c,
            mech_prod=mech_prod,
            vertical_start=gtx.int32(0),
            vertical_start_interior=gtx.int32(1),
            vertical_end=gtx.int32(grid.num_levels),
            cell_start_nudging=gtx.int32(cell_start_nudging),
            cell_start_lateral_boundary_level_3=gtx.int32(cell_start_lateral_boundary_level_3),
            cell_end_halo=gtx.int32(cell_end_halo),
        )


def stability_term_classic_numpy(
    mech_prod: np.ndarray, bruvais: np.ndarray, rturb_prandtl: float
) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, 0.5 * mech_prod - rturb_prandtl * bruvais))


def stability_term_louis_numpy(
    mech_prod: np.ndarray,
    bruvais: np.ndarray,
    scaling_factor_louis: np.ndarray,
    rturb_prandtl: float,
    louis_constant_b: float,
) -> np.ndarray:
    ri = 2.0 * bruvais / np.maximum(1.0e-28, mech_prod)  # eps_louis
    stability_function = np.maximum(
        1.0 - ri * rturb_prandtl,
        np.minimum(
            1.0,
            (1.0 / (1.0 + louis_constant_b * scaling_factor_louis[:, np.newaxis] * np.abs(ri)))
            ** 4.0,
        ),
    )
    return np.sqrt(0.5 * mech_prod * stability_function)


def compute_smagorinsky_viscosity_numpy(
    mech_prod: np.ndarray,
    bruvais: np.ndarray,
    rho_ic: np.ndarray,
    mixing_length_sq: np.ndarray,
    *,
    scaling_factor_louis: np.ndarray,
    fract_land: np.ndarray,
    fract_ice: np.ndarray,
    rturb_prandtl: float,
    louis_constant_b: float,
    use_louis: bool,
    use_louis_land: bool,
    use_louis_ice: bool,
) -> tuple[np.ndarray, np.ndarray]:
    nlev = mech_prod.shape[1] - 1

    if use_louis:
        # If the Louis formula is used but not over land and/or sea ice, use the
        # classic formulation for cells with more than 50% land fraction or more
        # than 50% ice fraction.
        classic_mask = ((not use_louis_land) & (fract_land > 0.5)) | (
            (not use_louis_ice) & (fract_ice > 0.5)
        )
        stability_term = np.where(
            classic_mask[:, np.newaxis],
            stability_term_classic_numpy(mech_prod, bruvais, rturb_prandtl),
            stability_term_louis_numpy(
                mech_prod, bruvais, scaling_factor_louis, rturb_prandtl, louis_constant_b
            ),
        )
    else:
        stability_term = stability_term_classic_numpy(mech_prod, bruvais, rturb_prandtl)

    km_ic = np.zeros_like(mech_prod)
    # interior half levels, Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1 (0-based)
    km_ic[:, 1:nlev] = rho_ic[:, 1:nlev] * mixing_length_sq[:, 1:nlev] * stability_term[:, 1:nlev]
    # boundary rows are copies of the adjacent interior rows
    # (Fortran 1-based: k = 1 <- k = 2, k = nlevp1 <- k = nlev)
    km_ic[:, 0] = km_ic[:, 1]
    km_ic[:, nlev] = km_ic[:, nlev - 1]
    kh_ic = km_ic * rturb_prandtl
    return km_ic, kh_ic


def smagorinsky_viscosity_reference(
    grid: base.Grid,
    *,
    mech_prod: np.ndarray,
    bruvais: np.ndarray,
    rho_ic: np.ndarray,
    mixing_length_sq: np.ndarray,
    scaling_factor_louis: np.ndarray,
    fract_land: np.ndarray,
    fract_ice: np.ndarray,
    rturb_prandtl: float,
    louis_constant_b: float,
    use_louis: bool,
    use_louis_land: bool,
    use_louis_ice: bool,
    **kwargs,
) -> dict:
    km_ic, kh_ic = compute_smagorinsky_viscosity_numpy(
        mech_prod,
        bruvais,
        rho_ic,
        mixing_length_sq,
        scaling_factor_louis=scaling_factor_louis,
        fract_land=fract_land,
        fract_ice=fract_ice,
        rturb_prandtl=rturb_prandtl,
        louis_constant_b=louis_constant_b,
        use_louis=use_louis,
        use_louis_land=use_louis_land,
        use_louis_ice=use_louis_ice,
    )
    return dict(km_ic=km_ic, kh_ic=kh_ic)


def smagorinsky_viscosity_input_data(
    data_alloc: stencil_tests.DataAllocationWrapper,
    grid: base.Grid,
    use_louis: bool,
    use_louis_land: bool,
    use_louis_ice: bool,
) -> dict[str, gtx.Field | state_utils.ScalarType]:
    mech_prod = data_alloc.random_field(
        dims.CellDim, dims.KHalfDim, low=0.0, high=0.01, dtype=wpfloat
    )
    bruvais = data_alloc.random_field(
        dims.CellDim, dims.KHalfDim, low=-0.001, high=0.001, dtype=wpfloat
    )
    rho_ic = data_alloc.random_field(dims.CellDim, dims.KHalfDim, low=0.5, high=1.4, dtype=wpfloat)
    mixing_length_sq = data_alloc.random_field(
        dims.CellDim, dims.KHalfDim, low=0.0, high=10000.0, dtype=wpfloat
    )
    scaling_factor_louis = data_alloc.random_field(dims.CellDim, low=0.5, high=2.0, dtype=wpfloat)
    fract_land = data_alloc.random_field(dims.CellDim, low=0.0, high=1.0, dtype=wpfloat)
    fract_ice = data_alloc.random_field(dims.CellDim, low=0.0, high=1.0, dtype=wpfloat)
    km_ic = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat)
    kh_ic = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat)

    return dict(
        mech_prod=mech_prod,
        bruvais=bruvais,
        rho_ic=rho_ic,
        mixing_length_sq=mixing_length_sq,
        scaling_factor_louis=scaling_factor_louis,
        fract_land=fract_land,
        fract_ice=fract_ice,
        km_ic=km_ic,
        kh_ic=kh_ic,
        rturb_prandtl=wpfloat(2.0),
        louis_constant_b=wpfloat(5.3),
        use_louis=use_louis,
        use_louis_land=use_louis_land,
        use_louis_ice=use_louis_ice,
        nlev=gtx.int32(grid.num_levels),
        horizontal_start=0,
        horizontal_end=gtx.int32(grid.num_cells),
        vertical_start=0,
        vertical_end=gtx.int32(grid.num_levels + 1),
    )


# Static-params variants: prove that the config bools can be passed both as regular
# runtime scalars ("none") and as static (compile-time) arguments selecting the variant.
SMAGORINSKY_VISCOSITY_STATIC_VARIANTS = {
    "none": (),
    "compile_time_variant": ("use_louis", "use_louis_land", "use_louis_ice"),
}


class TestComputeSmagorinskyViscosityClassic(stencil_tests.StencilTest):
    PROGRAM = compute_smagorinsky_viscosity
    OUTPUTS = ("km_ic", "kh_ic")
    STATIC_PARAMS = SMAGORINSKY_VISCOSITY_STATIC_VARIANTS

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return smagorinsky_viscosity_reference(grid, **kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return smagorinsky_viscosity_input_data(
            data_alloc, grid, use_louis=False, use_louis_land=True, use_louis_ice=True
        )


class TestComputeSmagorinskyViscosityLouis(stencil_tests.StencilTest):
    PROGRAM = compute_smagorinsky_viscosity
    OUTPUTS = ("km_ic", "kh_ic")
    STATIC_PARAMS = SMAGORINSKY_VISCOSITY_STATIC_VARIANTS

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return smagorinsky_viscosity_reference(grid, **kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return smagorinsky_viscosity_input_data(
            data_alloc, grid, use_louis=True, use_louis_land=True, use_louis_ice=True
        )


class TestComputeSmagorinskyViscosityLouisMaskedLandIce(stencil_tests.StencilTest):
    PROGRAM = compute_smagorinsky_viscosity
    OUTPUTS = ("km_ic", "kh_ic")
    STATIC_PARAMS = SMAGORINSKY_VISCOSITY_STATIC_VARIANTS

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return smagorinsky_viscosity_reference(grid, **kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return smagorinsky_viscosity_input_data(
            data_alloc, grid, use_louis=True, use_louis_land=False, use_louis_ice=False
        )


class TestComputeSmagorinskyViscosityLouisMaskedLandOnly(stencil_tests.StencilTest):
    PROGRAM = compute_smagorinsky_viscosity
    OUTPUTS = ("km_ic", "kh_ic")
    STATIC_PARAMS = SMAGORINSKY_VISCOSITY_STATIC_VARIANTS

    @stencil_tests.static_reference
    def reference(grid: base.Grid, **kwargs: Any) -> dict:
        return smagorinsky_viscosity_reference(grid, **kwargs)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        return smagorinsky_viscosity_input_data(
            data_alloc, grid, use_louis=True, use_louis_land=False, use_louis_ice=True
        )


def assign_constant_viscosity_numpy(
    rho_ic: np.ndarray, km_const: float, rturb_prandtl: float
) -> tuple[np.ndarray, np.ndarray]:
    nlev = rho_ic.shape[1] - 1
    km_ic = np.zeros_like(rho_ic)
    # interior half levels, Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1 (0-based)
    km_ic[:, 1:nlev] = rho_ic[:, 1:nlev] * km_const
    # boundary rows are copies of the adjacent interior rows
    # (Fortran 1-based: k = 1 <- k = 2, k = nlevp1 <- k = nlev)
    km_ic[:, 0] = km_ic[:, 1]
    km_ic[:, nlev] = km_ic[:, nlev - 1]
    kh_ic = km_ic * rturb_prandtl
    return km_ic, kh_ic


class TestAssignConstantViscosity(stencil_tests.StencilTest):
    PROGRAM = assign_constant_viscosity
    OUTPUTS = ("km_ic", "kh_ic")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        rho_ic: np.ndarray,
        km_const: float,
        rturb_prandtl: float,
        **kwargs,
    ) -> dict:
        km_ic, kh_ic = assign_constant_viscosity_numpy(rho_ic, km_const, rturb_prandtl)
        return dict(km_ic=km_ic, kh_ic=kh_ic)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        rho_ic = data_alloc.random_field(
            dims.CellDim, dims.KHalfDim, low=0.5, high=1.4, dtype=wpfloat
        )
        km_ic = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat)
        kh_ic = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat)

        return dict(
            rho_ic=rho_ic,
            km_ic=km_ic,
            kh_ic=kh_ic,
            km_const=wpfloat(0.05),
            rturb_prandtl=wpfloat(2.0),
            nlev=gtx.int32(grid.num_levels),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels + 1),
        )


def interpolate_km_to_full_level_cells_numpy(km_ic: np.ndarray, *, km_min: float) -> np.ndarray:
    return np.maximum(km_min, 0.5 * (km_ic[:, :-1] + km_ic[:, 1:]))


def interpolate_km_to_vertices_numpy(
    km_ic: np.ndarray, *, cells_aw_verts: np.ndarray, v2c: np.ndarray, km_min: float
) -> np.ndarray:
    return np.maximum(km_min, np.sum(cells_aw_verts[:, :, np.newaxis] * km_ic[v2c], axis=1))


def interpolate_km_to_edges_numpy(
    km_ic: np.ndarray, *, c_lin_e: np.ndarray, e2c: np.ndarray, km_min: float
) -> np.ndarray:
    return np.maximum(km_min, np.sum(km_ic[e2c] * c_lin_e[:, :, np.newaxis], axis=1))


@pytest.mark.skip_value_error
class TestInterpolateKm(stencil_tests.StencilTest):
    PROGRAM = interpolate_km
    OUTPUTS = ("km_c", "km_iv", "km_ie")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        km_ic: np.ndarray,
        cells_aw_verts: np.ndarray,
        c_lin_e: np.ndarray,
        km_min: float,
        **kwargs,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        return dict(
            km_c=interpolate_km_to_full_level_cells_numpy(km_ic, km_min=km_min),
            km_iv=interpolate_km_to_vertices_numpy(
                km_ic,
                cells_aw_verts=cells_aw_verts,
                v2c=connectivities[dims.V2C],
                km_min=km_min,
            ),
            km_ie=interpolate_km_to_edges_numpy(
                km_ic,
                c_lin_e=c_lin_e,
                e2c=connectivities[dims.E2C],
                km_min=km_min,
            ),
        )

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        km_ic = data_alloc.random_field(
            dims.CellDim, dims.KHalfDim, low=0.0, high=1.0, dtype=wpfloat
        )
        cells_aw_verts = data_alloc.random_field(
            dims.VertexDim, dims.V2CDim, low=0.0, high=1.0 / 6.0, dtype=wpfloat
        )
        c_lin_e = data_alloc.random_field(
            dims.EdgeDim, dims.E2CDim, low=0.0, high=1.0, dtype=wpfloat
        )

        return dict(
            km_ic=km_ic,
            cells_aw_verts=cells_aw_verts,
            c_lin_e=c_lin_e,
            km_c=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            km_iv=data_alloc.zero_field(dims.VertexDim, dims.KHalfDim, dtype=wpfloat),
            km_ie=data_alloc.zero_field(dims.EdgeDim, dims.KHalfDim, dtype=wpfloat),
            # large enough that the floor is active for part of each field
            km_min=wpfloat(0.5),
            cell_start=0,
            cell_end=gtx.int32(grid.num_cells),
            vertex_start=0,
            vertex_end=gtx.int32(grid.num_vertices),
            edge_start=0,
            edge_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
            vertical_end_half=gtx.int32(grid.num_levels + 1),
        )
