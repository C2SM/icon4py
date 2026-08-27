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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.diagnostics import (
    compute_edge_shear_diagnostics,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests


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
        dtype=ta.wpfloat,
        allocator=data_alloc.allocator,
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
        edge_end_end: int,
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
                (edge_start_lateral_boundary_level_2, edge_end_end),
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
        w = data_alloc.random_field(
            dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        vn = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        u_vert = data_alloc.random_field(dims.VertexDim, dims.KDim, dtype=ta.wpfloat)
        v_vert = data_alloc.random_field(dims.VertexDim, dims.KDim, dtype=ta.wpfloat)
        w_vert = data_alloc.random_field(
            dims.VertexDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )

        c_lin_e = data_alloc.random_field(dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat)
        wgtfac_e = data_alloc.random_field(
            dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
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

        w_ie = data_alloc.zero_field(
            dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        vn_ie = data_alloc.zero_field(
            dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        vt_ie = data_alloc.zero_field(
            dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
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
        edge_end_end = grid.end_index(edge_domain(h_grid.Zone.END))
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
            edge_end_end=gtx.int32(edge_end_end),
        )
