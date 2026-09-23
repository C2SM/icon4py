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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.wind_diffusion import (
    compute_vn_diffusion_tendency,
    compute_w_diffusion_tendency_and_update,
    interpolate_and_update_horizontal_wind,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests

from .test_vertical_diffusion import (
    diffusion_matrix_numpy,
    implicit_diffusion_tendency_numpy,
    matrix_diagonals_on_rows,
)


def on_rows(values: np.ndarray, rows: slice, levels: slice) -> np.ndarray:
    out = np.zeros_like(values)
    out[rows, levels] = values[rows, levels]
    return out


def edge_projection(field: np.ndarray, neighbors: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Projection of the vector field (field_x, field_y) at each neighbor onto (x, y)."""
    field_x, field_y = field
    return field_x[neighbors] * x[..., np.newaxis] + field_y[neighbors] * y[..., np.newaxis]


class TestComputeVnDiffusionTendency(stencil_tests.StencilTest):
    PROGRAM = compute_vn_diffusion_tendency
    OUTPUTS = ("vn_tendency",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        rho: np.ndarray,
        w: np.ndarray,
        vn: np.ndarray,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        km_c: np.ndarray,
        div_c: np.ndarray,
        km_iv: np.ndarray,
        km_ie: np.ndarray,
        u_stress: np.ndarray,
        v_stress: np.ndarray,
        inv_ddqz_z_full_e: np.ndarray,
        inv_ddqz_z_half_e: np.ndarray,
        c_lin_e: np.ndarray,
        primal_normal_cell_x: np.ndarray,
        primal_normal_cell_y: np.ndarray,
        primal_normal_vert_x: np.ndarray,
        primal_normal_vert_y: np.ndarray,
        dual_normal_vert_x: np.ndarray,
        dual_normal_vert_y: np.ndarray,
        tangent_orientation: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        dtime: float,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        e2c = connectivities[dims.E2C]
        e2c2v = connectivities[dims.E2C2V]
        nlev = vn.shape[1]
        tangent = tangent_orientation[:, np.newaxis]
        inv_primal = inv_primal_edge_length[:, np.newaxis]
        inv_vert_vert = inv_vert_vert_length[:, np.newaxis]
        inv_dual = inv_dual_edge_length[:, np.newaxis]

        inv_rhoe = 1.0 / np.sum(rho[e2c] * c_lin_e[..., np.newaxis], axis=1)

        vn_vert = edge_projection(
            (u_vert, v_vert), e2c2v, primal_normal_vert_x, primal_normal_vert_y
        )
        vt_vert = edge_projection((u_vert, v_vert), e2c2v, dual_normal_vert_x, dual_normal_vert_y)
        dvt = vt_vert[:, 3] - vt_vert[:, 2]
        flux_c = [
            km_c[e2c[:, i]]
            * (4.0 * sign * (vn_vert[:, 2 + i] - vn) * inv_vert_vert - 2.0 / 3.0 * div_c[e2c[:, i]])
            for i, sign in ((0, -1.0), (1, 1.0))
        ]
        km_v = km_iv[e2c2v[:, :2], :-1] + km_iv[e2c2v[:, :2], 1:]
        flux_v = [
            km_v[:, i]
            * (tangent * sign * (vn_vert[:, i] - vn) * inv_primal + 0.5 * dvt * inv_vert_vert)
            for i, sign in ((0, -1.0), (1, 1.0))
        ]
        horizontal_tendency = (
            (flux_c[1] - flux_c[0]) * inv_dual
            + 2.0 * tangent * (flux_v[1] - flux_v[0]) * inv_primal
        ) * inv_rhoe

        inv_air_mass = inv_ddqz_z_full_e * inv_rhoe
        dwdn_flux = km_ie * inv_dual * (w[e2c[:, 1]] - w[e2c[:, 0]])
        dwdn_flux[:, 0] = 0.0
        stress = edge_projection(
            (u_stress[:, np.newaxis], v_stress[:, np.newaxis]),
            e2c,
            primal_normal_cell_x,
            primal_normal_cell_y,
        )[..., 0]
        dwdn_flux[:, nlev] = np.sum(stress * c_lin_e, axis=1)
        rhs = (dwdn_flux[:, :-1] - dwdn_flux[:, 1:]) * inv_air_mass

        matrix = diffusion_matrix_numpy(
            km_ie[:, 1:nlev] * inv_ddqz_z_half_e[:, 1:nlev], inv_air_mass
        )
        rows = slice(0, nlev)
        a, b, c = matrix_diagonals_on_rows(matrix, vn.shape, rows)
        tendency = implicit_diffusion_tendency_numpy(
            var=vn, a=a, b=b, c=c, rhs=rhs, tend=horizontal_tendency, dtime=dtime, rows=rows
        )
        return dict(
            vn_tendency=on_rows(
                tendency,
                slice(horizontal_start, horizontal_end),
                slice(vertical_start, vertical_end),
            )
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        # narrower than the field, which the grid's zones are not
        horizontal_start, horizontal_end = 1, grid.num_edges - 1
        return dict(
            rho=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.5, high=2.0),
            w=data_alloc.random_field(dims.CellDim, dims.KHalfDim),
            vn=data_alloc.random_field(dims.EdgeDim, dims.KDim),
            u_vert=data_alloc.random_field(dims.VertexDim, dims.KDim),
            v_vert=data_alloc.random_field(dims.VertexDim, dims.KDim),
            km_c=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0),
            div_c=data_alloc.random_field(dims.CellDim, dims.KDim),
            km_iv=data_alloc.random_field(dims.VertexDim, dims.KHalfDim, low=0.0),
            km_ie=data_alloc.random_field(dims.EdgeDim, dims.KHalfDim, low=0.0),
            u_stress=data_alloc.random_field(dims.CellDim),
            v_stress=data_alloc.random_field(dims.CellDim),
            inv_ddqz_z_full_e=data_alloc.random_field(dims.EdgeDim, dims.KDim, low=0.1),
            inv_ddqz_z_half_e=data_alloc.random_field(dims.EdgeDim, dims.KHalfDim, low=0.1),
            c_lin_e=data_alloc.random_field(dims.EdgeDim, dims.E2CDim, low=0.1, high=0.9),
            primal_normal_cell_x=data_alloc.random_field(dims.EdgeDim, dims.E2CDim),
            primal_normal_cell_y=data_alloc.random_field(dims.EdgeDim, dims.E2CDim),
            primal_normal_vert_x=data_alloc.random_field(dims.EdgeDim, dims.E2C2VDim),
            primal_normal_vert_y=data_alloc.random_field(dims.EdgeDim, dims.E2C2VDim),
            dual_normal_vert_x=data_alloc.random_field(dims.EdgeDim, dims.E2C2VDim),
            dual_normal_vert_y=data_alloc.random_field(dims.EdgeDim, dims.E2C2VDim),
            tangent_orientation=data_alloc.random_sign(dims.EdgeDim, dtype=wpfloat),
            inv_primal_edge_length=data_alloc.random_field(dims.EdgeDim, low=0.1),
            inv_vert_vert_length=data_alloc.random_field(dims.EdgeDim, low=0.1),
            inv_dual_edge_length=data_alloc.random_field(dims.EdgeDim, low=0.1),
            vn_tendency=data_alloc.zero_field(dims.EdgeDim, dims.KDim),
            dtime=wpfloat(0.5),
            horizontal_start=gtx.int32(horizontal_start),
            horizontal_end=gtx.int32(horizontal_end),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )


class TestInterpolateAndUpdateHorizontalWind(stencil_tests.StencilTest):
    PROGRAM = interpolate_and_update_horizontal_wind
    OUTPUTS = ("tend_u", "tend_v", "new_u", "new_v")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        vn_tendency: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        rbf_coeff_c1: np.ndarray,
        rbf_coeff_c2: np.ndarray,
        dtime: float,
        tendency_horizontal_start: int,
        update_horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        c2e2c2e = stencil_tests.connectivities_asnumpy(grid)[dims.C2E2C2E]
        tend_u = np.sum(rbf_coeff_c1[..., np.newaxis] * vn_tendency[c2e2c2e], axis=1)
        tend_v = np.sum(rbf_coeff_c2[..., np.newaxis] * vn_tendency[c2e2c2e], axis=1)
        levels = slice(vertical_start, vertical_end)
        tendency_rows = slice(tendency_horizontal_start, horizontal_end)
        update_rows = slice(update_horizontal_start, horizontal_end)
        return dict(
            tend_u=on_rows(tend_u, tendency_rows, levels),
            tend_v=on_rows(tend_v, tendency_rows, levels),
            new_u=on_rows(u + tend_u * dtime, update_rows, levels),
            new_v=on_rows(v + tend_v * dtime, update_rows, levels),
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        tendency_horizontal_start, update_horizontal_start = 1, 2
        horizontal_end = grid.num_cells - 1
        return dict(
            vn_tendency=data_alloc.random_field(dims.EdgeDim, dims.KDim),
            u=data_alloc.random_field(dims.CellDim, dims.KDim),
            v=data_alloc.random_field(dims.CellDim, dims.KDim),
            rbf_coeff_c1=data_alloc.random_field(dims.CellDim, dims.C2E2C2EDim),
            rbf_coeff_c2=data_alloc.random_field(dims.CellDim, dims.C2E2C2EDim),
            tend_u=data_alloc.zero_field(dims.CellDim, dims.KDim),
            tend_v=data_alloc.zero_field(dims.CellDim, dims.KDim),
            new_u=data_alloc.zero_field(dims.CellDim, dims.KDim),
            new_v=data_alloc.zero_field(dims.CellDim, dims.KDim),
            dtime=wpfloat(0.5),
            tendency_horizontal_start=gtx.int32(tendency_horizontal_start),
            update_horizontal_start=gtx.int32(update_horizontal_start),
            horizontal_end=gtx.int32(horizontal_end),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )


class TestComputeWDiffusionTendencyAndUpdate(stencil_tests.StencilTest):
    PROGRAM = compute_w_diffusion_tendency_and_update
    OUTPUTS = ("horizontal_stress_tendency", "tend_w", "new_w")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        w: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        vn: np.ndarray,
        rho_ic: np.ndarray,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        w_vert: np.ndarray,
        w_ie: np.ndarray,
        km_c: np.ndarray,
        km_ic: np.ndarray,
        km_iv: np.ndarray,
        div_c: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        inv_ddqz_z_half: np.ndarray,
        inv_ddqz_z_half_v: np.ndarray,
        e_bln_c_s: np.ndarray,
        rbf_coeff_e: np.ndarray,
        primal_normal_cell_x: np.ndarray,
        primal_normal_cell_y: np.ndarray,
        dual_normal_vert_x: np.ndarray,
        dual_normal_vert_y: np.ndarray,
        edge_cell_length: np.ndarray,
        tangent_orientation: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        new_w: np.ndarray,
        dtime: float,
        edge_start: int,
        edge_end: int,
        cell_start: int,
        cell_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        e2c = connectivities[dims.E2C]
        e2c2v = connectivities[dims.E2C2V]
        e2c2e = connectivities[dims.E2C2E]
        c2e = connectivities[dims.C2E]
        nlev = u.shape[1]
        above, below = slice(0, nlev - 1), slice(1, nlev)
        interior = slice(1, nlev)
        tangent = tangent_orientation[:, np.newaxis]

        vt_e = np.sum(rbf_coeff_e[..., np.newaxis] * vn[e2c2e], axis=1)
        dvn = edge_projection((u, v), e2c, primal_normal_cell_x, primal_normal_cell_y)
        dvn = dvn[..., above] - dvn[..., below]
        vt_vert = edge_projection(
            (u_vert, v_vert), e2c2v[:, :2], dual_normal_vert_x[:, :2], dual_normal_vert_y[:, :2]
        )
        vt_mid = 0.5 * (vt_vert + vt_e[:, np.newaxis])
        dvt = vt_mid[..., above] - vt_mid[..., below]
        w_ie_interior = w_ie[:, interior]
        flux_c = [
            km_ic[e2c[:, i], interior]
            * (
                dvn[:, i] * inv_ddqz_z_half[e2c[:, i], interior]
                + sign
                * (w_vert[e2c2v[:, 2 + i], interior] - w_ie_interior)
                * 2.0
                * inv_vert_vert_length[:, np.newaxis]
            )
            for i, sign in ((0, -1.0), (1, 1.0))
        ]
        flux_v = [
            km_iv[e2c2v[:, i], interior]
            * (
                dvt[:, i] * inv_ddqz_z_half_v[e2c2v[:, i], interior]
                + tangent
                * sign
                * (w_vert[e2c2v[:, i], interior] - w_ie_interior)
                / edge_cell_length[:, i, np.newaxis]
            )
            for i, sign in ((0, -1.0), (1, 1.0))
        ]
        stress_tendency = np.zeros_like(w_ie)
        stress_tendency[:, interior] = (flux_c[1] - flux_c[0]) * inv_dual_edge_length[
            :, np.newaxis
        ] + (flux_v[1] - flux_v[0]) * tangent * 2.0 * inv_primal_edge_length[:, np.newaxis]
        edges = slice(edge_start, edge_end)
        stress_tendency = on_rows(stress_tendency, edges, slice(None))

        inv_rho_ic = 1.0 / rho_ic
        horizontal_tendency = inv_rho_ic * np.sum(
            e_bln_c_s[..., np.newaxis] * stress_tendency[c2e], axis=1
        )
        inv_air_mass = inv_rho_ic * inv_ddqz_z_half
        rhs = np.zeros_like(w)
        rhs[:, interior] = (
            2.0
            * inv_air_mass[:, interior]
            * (km_c[:, below] * div_c[:, below] - km_c[:, above] * div_c[:, above])
            / 3.0
        )
        # w = 0 on the top and bottom half levels: diffuse over all half levels and keep the
        # interior block, which then carries the boundary fluxes on its diagonal
        matrix = diffusion_matrix_numpy(2.0 * km_c * inv_ddqz_z_full, inv_air_mass)
        a, b, c = matrix_diagonals_on_rows(matrix[:, interior, interior], w.shape, interior)
        tend_w = implicit_diffusion_tendency_numpy(
            var=w, a=a, b=b, c=c, rhs=rhs, tend=horizontal_tendency, dtime=dtime, rows=interior
        )

        cells = slice(cell_start, cell_end)
        levels = slice(vertical_start, vertical_end)
        # the diffused rows, zero on the two bounding half levels, and the input elsewhere
        expected_new_w = new_w.copy()
        expected_new_w[cells, vertical_start - 1 : vertical_end + 1] = 0.0
        expected_new_w[cells, levels] = (w + tend_w * dtime)[cells, levels]
        return dict(
            horizontal_stress_tendency=stress_tendency,
            tend_w=on_rows(tend_w, cells, levels),
            new_w=expected_new_w,
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        # narrower than the fields and different per dimension, which the grid's zones are not
        edge_start, edge_end = 1, grid.num_edges - 1
        cell_start, cell_end = 2, grid.num_cells - 2
        return dict(
            w=data_alloc.random_field(dims.CellDim, dims.KHalfDim),
            u=data_alloc.random_field(dims.CellDim, dims.KDim),
            v=data_alloc.random_field(dims.CellDim, dims.KDim),
            vn=data_alloc.random_field(dims.EdgeDim, dims.KDim),
            rho_ic=data_alloc.random_field(dims.CellDim, dims.KHalfDim, low=0.5, high=2.0),
            u_vert=data_alloc.random_field(dims.VertexDim, dims.KDim),
            v_vert=data_alloc.random_field(dims.VertexDim, dims.KDim),
            w_vert=data_alloc.random_field(dims.VertexDim, dims.KHalfDim),
            w_ie=data_alloc.random_field(dims.EdgeDim, dims.KHalfDim),
            km_c=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0),
            km_ic=data_alloc.random_field(dims.CellDim, dims.KHalfDim, low=0.0),
            km_iv=data_alloc.random_field(dims.VertexDim, dims.KHalfDim, low=0.0),
            div_c=data_alloc.random_field(dims.CellDim, dims.KDim),
            inv_ddqz_z_full=data_alloc.random_field(dims.CellDim, dims.KDim, low=0.1),
            inv_ddqz_z_half=data_alloc.random_field(dims.CellDim, dims.KHalfDim, low=0.1),
            inv_ddqz_z_half_v=data_alloc.random_field(dims.VertexDim, dims.KHalfDim, low=0.1),
            e_bln_c_s=data_alloc.random_field(dims.CellDim, dims.C2EDim),
            rbf_coeff_e=data_alloc.random_field(dims.EdgeDim, dims.E2C2EDim),
            primal_normal_cell_x=data_alloc.random_field(dims.EdgeDim, dims.E2CDim),
            primal_normal_cell_y=data_alloc.random_field(dims.EdgeDim, dims.E2CDim),
            dual_normal_vert_x=data_alloc.random_field(dims.EdgeDim, dims.E2C2VDim),
            dual_normal_vert_y=data_alloc.random_field(dims.EdgeDim, dims.E2C2VDim),
            edge_cell_length=data_alloc.random_field(dims.EdgeDim, dims.E2CDim, low=0.1),
            tangent_orientation=data_alloc.random_sign(dims.EdgeDim, dtype=wpfloat),
            inv_primal_edge_length=data_alloc.random_field(dims.EdgeDim, low=0.1),
            inv_vert_vert_length=data_alloc.random_field(dims.EdgeDim, low=0.1),
            inv_dual_edge_length=data_alloc.random_field(dims.EdgeDim, low=0.1),
            horizontal_stress_tendency=data_alloc.zero_field(dims.EdgeDim, dims.KHalfDim),
            tend_w=data_alloc.zero_field(dims.CellDim, dims.KHalfDim),
            new_w=data_alloc.random_field(dims.CellDim, dims.KHalfDim),
            dtime=wpfloat(0.5),
            edge_start=gtx.int32(edge_start),
            edge_end=gtx.int32(edge_end),
            cell_start=gtx.int32(cell_start),
            cell_end=gtx.int32(cell_end),
            vertical_start=gtx.int32(1),
            vertical_end=gtx.int32(grid.num_levels),
        )
