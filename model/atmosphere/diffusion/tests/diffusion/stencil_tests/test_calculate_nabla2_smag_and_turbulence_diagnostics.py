# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_smag_and_turbulence_diagnostics import (
    calculate_nabla2_smag_and_turbulence_diagnostics,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests


def _nabla2_and_smag_coefficients_for_vn_numpy(
    connectivities: dict,
    *,
    diff_multfac_smag: np.ndarray,
    tangent_orientation: np.ndarray,
    inv_primal_edge_length: np.ndarray,
    inv_vert_vert_length: np.ndarray,
    u_vert: np.ndarray,
    v_vert: np.ndarray,
    primal_normal_vert_x: np.ndarray,
    primal_normal_vert_y: np.ndarray,
    dual_normal_vert_x: np.ndarray,
    dual_normal_vert_y: np.ndarray,
    vn: np.ndarray,
    smag_limit: np.ndarray,
    smag_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e2c2v = connectivities[dims.E2C2V]
    u_vert_e2c2v = u_vert[e2c2v]
    v_vert_e2c2v = v_vert[e2c2v]
    dual_normal_vert_x = np.expand_dims(dual_normal_vert_x, axis=-1)
    dual_normal_vert_y = np.expand_dims(dual_normal_vert_y, axis=-1)
    primal_normal_vert_x = np.expand_dims(primal_normal_vert_x, axis=-1)
    primal_normal_vert_y = np.expand_dims(primal_normal_vert_y, axis=-1)
    inv_vert_vert_length = np.expand_dims(inv_vert_vert_length, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)
    tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)

    dvt_tang = (
        -(
            u_vert_e2c2v[:, 0] * dual_normal_vert_x[:, 0]
            + v_vert_e2c2v[:, 0] * dual_normal_vert_y[:, 0]
        )
    ) + (
        u_vert_e2c2v[:, 1] * dual_normal_vert_x[:, 1]
        + v_vert_e2c2v[:, 1] * dual_normal_vert_y[:, 1]
    )

    dvt_norm = (
        -(
            u_vert_e2c2v[:, 2] * dual_normal_vert_x[:, 2]
            + v_vert_e2c2v[:, 2] * dual_normal_vert_y[:, 2]
        )
    ) + (
        u_vert_e2c2v[:, 3] * dual_normal_vert_x[:, 3]
        + v_vert_e2c2v[:, 3] * dual_normal_vert_y[:, 3]
    )

    kh_smag_1 = (
        -(
            u_vert_e2c2v[:, 0] * primal_normal_vert_x[:, 0]
            + v_vert_e2c2v[:, 0] * primal_normal_vert_y[:, 0]
        )
    ) + (
        u_vert_e2c2v[:, 1] * primal_normal_vert_x[:, 1]
        + v_vert_e2c2v[:, 1] * primal_normal_vert_y[:, 1]
    )

    dvt_tang = dvt_tang * tangent_orientation

    kh_smag_1 = (kh_smag_1 * tangent_orientation * inv_primal_edge_length) + (
        dvt_norm * inv_vert_vert_length
    )
    kh_smag_1 = kh_smag_1 * kh_smag_1

    kh_smag_2 = (
        -(
            u_vert_e2c2v[:, 2] * primal_normal_vert_x[:, 2]
            + v_vert_e2c2v[:, 2] * primal_normal_vert_y[:, 2]
        )
    ) + (
        u_vert_e2c2v[:, 3] * primal_normal_vert_x[:, 3]
        + v_vert_e2c2v[:, 3] * primal_normal_vert_y[:, 3]
    )
    kh_smag_2 = (kh_smag_2 * inv_vert_vert_length) - (dvt_tang * inv_primal_edge_length)
    kh_smag_2 = kh_smag_2 * kh_smag_2

    kh_smag_pre_limit = diff_multfac_smag * np.sqrt(kh_smag_2 + kh_smag_1)

    z_nabla2_e = (
        (
            (
                u_vert_e2c2v[:, 0] * primal_normal_vert_x[:, 0]
                + v_vert_e2c2v[:, 0] * primal_normal_vert_y[:, 0]
            )
            + (
                u_vert_e2c2v[:, 1] * primal_normal_vert_x[:, 1]
                + v_vert_e2c2v[:, 1] * primal_normal_vert_y[:, 1]
            )
        )
        - 2.0 * vn
    ) * (inv_primal_edge_length * inv_primal_edge_length)

    z_nabla2_e = z_nabla2_e + (
        (
            (
                u_vert_e2c2v[:, 2] * primal_normal_vert_x[:, 2]
                + v_vert_e2c2v[:, 2] * primal_normal_vert_y[:, 2]
            )
            + (
                u_vert_e2c2v[:, 3] * primal_normal_vert_x[:, 3]
                + v_vert_e2c2v[:, 3] * primal_normal_vert_y[:, 3]
            )
        )
        - 2.0 * vn
    ) * (inv_vert_vert_length * inv_vert_vert_length)

    z_nabla2_e = 4.0 * z_nabla2_e

    kh_smag_ec = kh_smag_pre_limit
    kh_smag_e = np.maximum(0.0, kh_smag_pre_limit - smag_offset)
    kh_smag_e = np.minimum(kh_smag_e, smag_limit)
    return kh_smag_e, kh_smag_ec, z_nabla2_e


def _diagnostic_quantities_for_turbulence_numpy(
    connectivities: dict,
    *,
    kh_smag_ec: np.ndarray,
    vn: np.ndarray,
    e_bln_c_s: np.ndarray,
    geofac_div: np.ndarray,
    diff_multfac_smag: np.ndarray,
    wgtfac_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c2e = connectivities[dims.C2E]
    e_bln_c_s_e = np.expand_dims(e_bln_c_s, axis=-1)
    geofac_div_e = np.expand_dims(geofac_div, axis=-1)
    diff_multfac_smag_e = np.expand_dims(diff_multfac_smag, axis=0)

    kh_c = np.sum(kh_smag_ec[c2e] * e_bln_c_s_e, axis=1) / diff_multfac_smag_e
    div = np.sum(vn[c2e] * geofac_div_e, axis=1)

    nlev = div.shape[1]
    w = wgtfac_c[:, 1:nlev]
    div_ic = np.zeros_like(wgtfac_c)
    hdef_ic = np.zeros_like(wgtfac_c)
    div_ic[:, 1:nlev] = w * div[:, 1:nlev] + (1.0 - w) * div[:, 0 : nlev - 1]
    hdef_ic[:, 1:nlev] = (w * kh_c[:, 1:nlev] + (1.0 - w) * kh_c[:, 0 : nlev - 1]) ** 2
    return div_ic, hdef_ic


@pytest.mark.continuous_benchmarking
class TestCalculateNabla2SmagAndTurbulenceDiagnostics(stencil_tests.StencilTest):
    PROGRAM = calculate_nabla2_smag_and_turbulence_diagnostics
    OUTPUTS = ("kh_smag_e", "z_nabla2_e", "div_ic", "hdef_ic")
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "compute_diagnostic_quantities",
            "edge_horizontal_start",
            "edge_horizontal_end",
            "edge_vertical_start",
            "edge_vertical_end",
            "cell_horizontal_start",
            "cell_horizontal_end",
            "cell_vertical_start",
            "cell_vertical_end",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "compute_diagnostic_quantities",
            "edge_vertical_start",
            "edge_vertical_end",
            "cell_vertical_start",
            "cell_vertical_end",
        ),
    }

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        diff_multfac_smag: np.ndarray,
        tangent_orientation: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        primal_normal_vert_x: np.ndarray,
        primal_normal_vert_y: np.ndarray,
        dual_normal_vert_x: np.ndarray,
        dual_normal_vert_y: np.ndarray,
        vn: np.ndarray,
        smag_limit: np.ndarray,
        e_bln_c_s: np.ndarray,
        geofac_div: np.ndarray,
        wgtfac_c: np.ndarray,
        kh_smag_e: np.ndarray,
        z_nabla2_e: np.ndarray,
        div_ic: np.ndarray,
        hdef_ic: np.ndarray,
        smag_offset: float,
        compute_diagnostic_quantities: bool,
        edge_horizontal_start: int,
        edge_horizontal_end: int,
        edge_vertical_start: int,
        edge_vertical_end: int,
        cell_horizontal_start: int,
        cell_horizontal_end: int,
        cell_vertical_start: int,
        cell_vertical_end: int,
        **kwargs,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)

        kh_smag_e_full, kh_smag_ec_full, z_nabla2_e_full = (
            _nabla2_and_smag_coefficients_for_vn_numpy(
                connectivities,
                diff_multfac_smag=diff_multfac_smag,
                tangent_orientation=tangent_orientation,
                inv_primal_edge_length=inv_primal_edge_length,
                inv_vert_vert_length=inv_vert_vert_length,
                u_vert=u_vert,
                v_vert=v_vert,
                primal_normal_vert_x=primal_normal_vert_x,
                primal_normal_vert_y=primal_normal_vert_y,
                dual_normal_vert_x=dual_normal_vert_x,
                dual_normal_vert_y=dual_normal_vert_y,
                vn=vn,
                smag_limit=smag_limit,
                smag_offset=smag_offset,
            )
        )

        kh_smag_e_out = kh_smag_e.copy()
        z_nabla2_e_out = z_nabla2_e.copy()
        kh_smag_e_out[
            edge_horizontal_start:edge_horizontal_end, edge_vertical_start:edge_vertical_end
        ] = kh_smag_e_full[
            edge_horizontal_start:edge_horizontal_end, edge_vertical_start:edge_vertical_end
        ]
        z_nabla2_e_out[
            edge_horizontal_start:edge_horizontal_end, edge_vertical_start:edge_vertical_end
        ] = z_nabla2_e_full[
            edge_horizontal_start:edge_horizontal_end, edge_vertical_start:edge_vertical_end
        ]

        div_ic_out = div_ic.copy()
        hdef_ic_out = hdef_ic.copy()
        if compute_diagnostic_quantities:
            div_ic_full, hdef_ic_full = _diagnostic_quantities_for_turbulence_numpy(
                connectivities,
                kh_smag_ec=kh_smag_ec_full,
                vn=vn,
                e_bln_c_s=e_bln_c_s,
                geofac_div=geofac_div,
                diff_multfac_smag=diff_multfac_smag,
                wgtfac_c=wgtfac_c,
            )
            div_ic_out[
                cell_horizontal_start:cell_horizontal_end,
                cell_vertical_start:cell_vertical_end,
            ] = div_ic_full[
                cell_horizontal_start:cell_horizontal_end,
                cell_vertical_start:cell_vertical_end,
            ]
            hdef_ic_out[
                cell_horizontal_start:cell_horizontal_end,
                cell_vertical_start:cell_vertical_end,
            ] = hdef_ic_full[
                cell_horizontal_start:cell_horizontal_end,
                cell_vertical_start:cell_vertical_end,
            ]

        return dict(
            kh_smag_e=kh_smag_e_out,
            z_nabla2_e=z_nabla2_e_out,
            div_ic=div_ic_out,
            hdef_ic=hdef_ic_out,
        )

    @stencil_tests.input_data_fixture(
        params=[{"compute_diagnostic_quantities": value} for value in [True, False]],
        ids=lambda param: (
            f"compute_diagnostic_quantities[{param['compute_diagnostic_quantities']}]"
        ),
    )
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper,
        grid: base.Grid,
        request: pytest.FixtureRequest,
    ) -> dict:
        compute_diagnostic_quantities = request.param["compute_diagnostic_quantities"]

        u_vert = data_alloc.random_field(dims.VertexDim, dims.KDim, dtype=ta.vpfloat)
        v_vert = data_alloc.random_field(dims.VertexDim, dims.KDim, dtype=ta.vpfloat)
        smag_offset = ta.vpfloat("9.0")
        diff_multfac_smag = data_alloc.random_field(dims.KDim, dtype=ta.vpfloat)
        tangent_orientation = data_alloc.random_sign(dims.EdgeDim, dtype=ta.wpfloat)
        vn = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        smag_limit = data_alloc.random_field(dims.KDim, dtype=ta.vpfloat)
        inv_vert_vert_length = data_alloc.random_field(dims.EdgeDim, dtype=ta.wpfloat)
        inv_primal_edge_length = data_alloc.random_field(dims.EdgeDim, dtype=ta.wpfloat)
        primal_normal_vert_x = data_alloc.random_field(
            dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        primal_normal_vert_y = data_alloc.random_field(
            dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        dual_normal_vert_x = data_alloc.random_field(dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat)
        dual_normal_vert_y = data_alloc.random_field(dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat)

        e_bln_c_s = data_alloc.random_field(dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)
        geofac_div = data_alloc.random_field(dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)
        wgtfac_c = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)

        kh_smag_e = data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        z_nabla2_e = data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        div_ic = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)
        hdef_ic = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)

        edge_domain = h_grid.domain(dims.EdgeDim)
        cell_domain = h_grid.domain(dims.CellDim)
        edge_horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
        edge_horizontal_end = grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
        cell_horizontal_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        cell_horizontal_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        assert edge_horizontal_start < edge_horizontal_end
        assert cell_horizontal_start < cell_horizontal_end

        return dict(
            diff_multfac_smag=diff_multfac_smag,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            inv_vert_vert_length=inv_vert_vert_length,
            u_vert=u_vert,
            v_vert=v_vert,
            primal_normal_vert_x=primal_normal_vert_x,
            primal_normal_vert_y=primal_normal_vert_y,
            dual_normal_vert_x=dual_normal_vert_x,
            dual_normal_vert_y=dual_normal_vert_y,
            vn=vn,
            smag_limit=smag_limit,
            e_bln_c_s=e_bln_c_s,
            geofac_div=geofac_div,
            wgtfac_c=wgtfac_c,
            kh_smag_e=kh_smag_e,
            z_nabla2_e=z_nabla2_e,
            div_ic=div_ic,
            hdef_ic=hdef_ic,
            smag_offset=smag_offset,
            compute_diagnostic_quantities=compute_diagnostic_quantities,
            edge_horizontal_start=edge_horizontal_start,
            edge_horizontal_end=edge_horizontal_end,
            edge_vertical_start=gtx.int32(0),
            edge_vertical_end=gtx.int32(grid.num_levels),
            cell_horizontal_start=cell_horizontal_start,
            cell_horizontal_end=cell_horizontal_end,
            cell_vertical_start=gtx.int32(1),
            cell_vertical_end=gtx.int32(grid.num_levels),
        )
