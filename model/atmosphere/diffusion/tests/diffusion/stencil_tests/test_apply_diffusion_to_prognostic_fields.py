# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import itertools
import math
from collections.abc import Mapping

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_prognostic_fields import (
    apply_diffusion_to_prognostic_fields,
)
from icon4py.model.common import constants, dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests, test_utils

from .test_apply_nabla2_to_vn_in_lateral_boundary import (
    apply_nabla2_to_vn_in_lateral_boundary_numpy,
)


def update_theta_and_exner_numpy(
    z_temp: np.ndarray,
    area: np.ndarray,
    theta_v: np.ndarray,
    exner: np.ndarray,
    rd_o_cvd: float,
) -> tuple[np.ndarray, np.ndarray]:
    area = np.expand_dims(area, axis=-1)
    z_theta = theta_v
    theta_v = theta_v + (area * z_temp)
    exner = exner * (1.0 + rd_o_cvd * (theta_v / z_theta - 1.0))
    return theta_v, exner


def truly_horizontal_diffusion_nabla_of_theta_over_steep_points_numpy(
    *,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    zd_vertoffset: np.ndarray,
    zd_diffcoef: np.ndarray,
    geofac_n2s_c: np.ndarray,
    geofac_n2s_nbh: np.ndarray,
    vcoef: np.ndarray,
    theta_v: np.ndarray,
    z_temp: np.ndarray,
    **kwargs,
) -> np.ndarray:
    c2e2c = connectivities[dims.C2E2C]
    full_shape = vcoef.shape

    geofac_n2s_nbh = np.expand_dims(geofac_n2s_nbh, axis=2)

    theta_v_at_zd_vertidx = np.zeros_like(vcoef)
    theta_v_at_zd_vertidx_p1 = np.zeros_like(vcoef)
    for ic in range(full_shape[0]):
        for isparse in range(full_shape[1]):
            for ik in range(full_shape[2]):
                if zd_diffcoef[ic, ik] == 0.0:
                    continue
                theta_v_at_zd_vertidx[ic, isparse, ik] = theta_v[
                    c2e2c[ic, isparse], ik + zd_vertoffset[ic, isparse, ik]
                ]
                theta_v_at_zd_vertidx_p1[ic, isparse, ik] = theta_v[
                    c2e2c[ic, isparse], ik + zd_vertoffset[ic, isparse, ik] + 1
                ]

    sum_over = np.sum(
        geofac_n2s_nbh * (vcoef * theta_v_at_zd_vertidx + (1.0 - vcoef) * theta_v_at_zd_vertidx_p1),
        axis=1,
    )

    geofac_n2s_c = np.expand_dims(geofac_n2s_c, axis=1)
    return z_temp + zd_diffcoef * (theta_v * geofac_n2s_c + sum_over)


def apply_nabla2_to_w_numpy(
    *,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    area: np.ndarray,
    z_nabla2_c: np.ndarray,
    geofac_n2s: np.ndarray,
    w: np.ndarray,
    diff_multfac_w: float,
) -> np.ndarray:
    c2e2cO = connectivities[dims.C2E2CO]
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)
    area = np.expand_dims(area, axis=-1)
    w = w - diff_multfac_w * area * area * np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], z_nabla2_c[c2e2cO] * geofac_n2s, 0.0), axis=1
    )
    return w


def apply_nabla2_to_w_in_upper_damping_layer_numpy(
    w: np.ndarray,
    diff_multfac_n2w: np.ndarray,
    area: np.ndarray,
    z_nabla2_c: np.ndarray,
) -> np.ndarray:
    area = np.expand_dims(area, axis=-1)
    w = w + diff_multfac_n2w * area * z_nabla2_c
    return w


def calculate_nabla2_for_w_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray], w: np.ndarray, geofac_n2s: np.ndarray
) -> np.ndarray:
    c2e2cO = connectivities[dims.C2E2CO]
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)
    z_nabla2_c = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], w[c2e2cO] * geofac_n2s, 0), axis=1
    )
    return z_nabla2_c


def apply_nabla2_and_nabla4_global_to_vn_numpy(
    *,
    area_edge: np.ndarray,
    kh_smag_e: np.ndarray,
    z_nabla2_e: np.ndarray,
    z_nabla4_e2: np.ndarray,
    diff_multfac_vn: np.ndarray,
    vn: np.ndarray,
) -> np.ndarray:
    area_edge = np.expand_dims(area_edge, axis=-1)
    diff_multfac_vn = np.expand_dims(diff_multfac_vn, axis=0)
    vn = vn + area_edge * (kh_smag_e * z_nabla2_e - diff_multfac_vn * z_nabla4_e2 * area_edge)
    return vn


def apply_nabla2_and_nabla4_to_vn_numpy(
    *,
    area_edge: np.ndarray,
    kh_smag_e: np.ndarray,
    z_nabla2_e: np.ndarray,
    z_nabla4_e2: np.ndarray,
    diff_multfac_vn: np.ndarray,
    nudgecoeff_e: np.ndarray,
    vn: np.ndarray,
    nudgezone_diff: float,
) -> np.ndarray:
    area_edge = np.expand_dims(area_edge, axis=-1)
    diff_multfac_vn = np.expand_dims(diff_multfac_vn, axis=0)
    nudgecoeff_e = np.expand_dims(nudgecoeff_e, axis=-1)
    vn = vn + area_edge * (
        np.maximum(nudgezone_diff * nudgecoeff_e, kh_smag_e) * z_nabla2_e
        - diff_multfac_vn * z_nabla4_e2 * area_edge
    )
    return vn


def calculate_diagnostics_for_turbulence_numpy(
    wgtfac_c: np.ndarray,
    div: np.ndarray,
    kh_c: np.ndarray,
    div_ic: np.ndarray,
    hdef_ic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nlev = div.shape[1]
    w = wgtfac_c[:, 1:nlev]
    div_ic[:, 1:nlev] = w * div[:, 1:nlev] + (1.0 - w) * div[:, 0 : nlev - 1]
    hdef_ic[:, 1:nlev] = (w * kh_c[:, 1:nlev] + (1.0 - w) * kh_c[:, 0 : nlev - 1]) ** 2
    return div_ic, hdef_ic


def calculate_horizontal_gradients_for_turbulence_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    w: np.ndarray,
    geofac_grg_x: np.ndarray,
    geofac_grg_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c2e2cO = connectivities[dims.C2E2CO]
    geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
    dwdx = np.sum(np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_x * w[c2e2cO], 0.0), axis=1)

    geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
    dwdy = np.sum(np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_y * w[c2e2cO], 0.0), axis=1)
    return dwdx, dwdy


def calculate_nabla4_numpy(
    *,
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    u_vert: np.ndarray,
    v_vert: np.ndarray,
    primal_normal_vert_x: np.ndarray,
    primal_normal_vert_y: np.ndarray,
    z_nabla2_e: np.ndarray,
    inv_vert_vert_length: np.ndarray,
    inv_primal_edge_length: np.ndarray,
) -> np.ndarray:
    e2c2v = connectivities[dims.E2C2V]
    u_vert_e2c2v = u_vert[e2c2v]
    v_vert_e2c2v = v_vert[e2c2v]

    primal_normal_vert_x = np.expand_dims(primal_normal_vert_x, axis=-1)
    primal_normal_vert_y = np.expand_dims(primal_normal_vert_y, axis=-1)
    inv_vert_vert_length = np.expand_dims(inv_vert_vert_length, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)

    nabv_tang = (
        u_vert_e2c2v[:, 0] * primal_normal_vert_x[:, 0]
        + v_vert_e2c2v[:, 0] * primal_normal_vert_y[:, 0]
    ) + (
        u_vert_e2c2v[:, 1] * primal_normal_vert_x[:, 1]
        + v_vert_e2c2v[:, 1] * primal_normal_vert_y[:, 1]
    )
    nabv_norm = (
        u_vert_e2c2v[:, 2] * primal_normal_vert_x[:, 2]
        + v_vert_e2c2v[:, 2] * primal_normal_vert_y[:, 2]
    ) + (
        u_vert_e2c2v[:, 3] * primal_normal_vert_x[:, 3]
        + v_vert_e2c2v[:, 3] * primal_normal_vert_y[:, 3]
    )
    z_nabla4_e2 = 4.0 * (
        (nabv_norm - 2.0 * z_nabla2_e) * inv_vert_vert_length**2
        + (nabv_tang - 2.0 * z_nabla2_e) * inv_primal_edge_length**2
    )
    return z_nabla4_e2


@pytest.mark.single_precision_ready
@pytest.mark.uses_concat_where
@pytest.mark.continuous_benchmarking
class TestApplyDiffusionToPrognosticFields(stencil_tests.StencilTest):
    PROGRAM = apply_diffusion_to_prognostic_fields
    OUTPUTS = ("vn", "w", "dwdx", "dwdy", "theta_v", "exner", "div_ic", "hdef_ic")
    ABSOLUTE_TOLERANCE = 0.0 if test_utils.wp_is_dp else 3e-7
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "compute_flow_deformation",
            "apply_to_temperature",
            "limited_area",
            "compute_hgrad_w",
            "start_2nd_nudge_line_idx_e",
            "nrdmax",
            "nlev",
            "w_horizontal_start",
            "w_horizontal_end",
            "edge_horizontal_start",
            "edge_horizontal_end",
            "hgrad_w_horizontal_start",
            "hgrad_w_horizontal_end",
            "cell_horizontal_start",
            "cell_horizontal_end",
            "vertical_start",
            "vertical_end",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "compute_flow_deformation",
            "apply_to_temperature",
            "compute_hgrad_w",
            "vertical_start",
            "vertical_end",
            "nrdmax",
            "nlev",
        ),
    }

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        kh_smag: np.ndarray,
        smag_limit: np.ndarray,
        smag_offset: float,
        vn_in: np.ndarray,
        e_bln_c_s: np.ndarray,
        geofac_div: np.ndarray,
        diff_multfac_smag: np.ndarray,
        wgtfac_c: np.ndarray,
        div_ic: np.ndarray,
        hdef_ic: np.ndarray,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        primal_normal_vert_x: np.ndarray,
        primal_normal_vert_y: np.ndarray,
        z_nabla2_e: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        area_edge: np.ndarray,
        diff_multfac_vn: np.ndarray,
        nudgecoeff_e: np.ndarray,
        nudgezone_diff: float,
        fac_bdydiff_v: float,
        area: np.ndarray,
        geofac_n2s: np.ndarray,
        geofac_grg_x: np.ndarray,
        geofac_grg_y: np.ndarray,
        w_old: np.ndarray,
        compute_hgrad_w: bool,
        dwdx: np.ndarray,
        dwdy: np.ndarray,
        diff_multfac_w: float,
        diff_multfac_n2w: np.ndarray,
        theta_v_in: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        zd_vertoffset: np.ndarray,
        zd_diffcoef: np.ndarray,
        geofac_n2s_c: np.ndarray,
        geofac_n2s_nbh: np.ndarray,
        vcoef: np.ndarray,
        theta_v: np.ndarray,
        exner: np.ndarray,
        vn: np.ndarray,
        theta_ref_mc: np.ndarray,
        thresh_tdiff: float,
        smallest_vpfloat: float,
        rd_o_cvd: float,
        apply_zdiffusion_t: bool,
        compute_flow_deformation: bool,
        apply_to_temperature: bool,
        limited_area: bool,
        start_2nd_nudge_line_idx_e: int,
        nrdmax: int,
        nlev: int,
        w_horizontal_start: int,
        w_horizontal_end: int,
        edge_horizontal_start: int,
        edge_horizontal_end: int,
        hgrad_w_horizontal_start: int,
        hgrad_w_horizontal_end: int,
        cell_horizontal_start: int,
        cell_horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        c2e = connectivities[dims.C2E]
        c2e2c = connectivities[dims.C2E2C]
        e2c = connectivities[dims.E2C]
        edge = np.arange(area_edge.shape[0])
        reshaped_k = np.arange(w_old.shape[1])[np.newaxis, :]

        # Smagorinsky limiter
        kh_smag_limited = np.maximum(0.0, kh_smag - smag_offset)
        kh_smag_limited = np.minimum(kh_smag_limited, smag_limit)

        # Turbulence diagnostics from pre-diffusion vn
        out_div_ic = div_ic.copy()
        out_hdef_ic = hdef_ic.copy()
        if compute_flow_deformation:
            e_bln_c_s_e = np.expand_dims(e_bln_c_s, axis=-1)
            geofac_div_e = np.expand_dims(geofac_div, axis=-1)
            diff_multfac_smag_e = np.expand_dims(diff_multfac_smag, axis=0)
            kh_c = np.sum(kh_smag[c2e] * e_bln_c_s_e, axis=1) / diff_multfac_smag_e
            div = np.sum(vn_in[c2e] * geofac_div_e, axis=1)
            div_ic_full, hdef_ic_full = calculate_diagnostics_for_turbulence_numpy(
                wgtfac_c, div, kh_c, np.zeros_like(div_ic), np.zeros_like(hdef_ic)
            )
            diag_slice = (
                slice(cell_horizontal_start, cell_horizontal_end),
                slice(vertical_start + 1, vertical_end),
            )
            out_div_ic[diag_slice] = div_ic_full[diag_slice]
            out_hdef_ic[diag_slice] = hdef_ic_full[diag_slice]

        # Diffusion of vn
        z_nabla4_e2 = calculate_nabla4_numpy(
            connectivities=connectivities,
            u_vert=u_vert,
            v_vert=v_vert,
            primal_normal_vert_x=primal_normal_vert_x,
            primal_normal_vert_y=primal_normal_vert_y,
            z_nabla2_e=z_nabla2_e,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_primal_edge_length=inv_primal_edge_length,
        )
        condition = start_2nd_nudge_line_idx_e <= edge[:, np.newaxis]
        if limited_area:
            vn_new = np.where(
                condition,
                apply_nabla2_and_nabla4_to_vn_numpy(
                    area_edge=area_edge,
                    kh_smag_e=kh_smag_limited,
                    z_nabla2_e=z_nabla2_e,
                    z_nabla4_e2=z_nabla4_e2,
                    diff_multfac_vn=diff_multfac_vn,
                    nudgecoeff_e=nudgecoeff_e,
                    vn=vn_in,
                    nudgezone_diff=nudgezone_diff,
                ),
                apply_nabla2_to_vn_in_lateral_boundary_numpy(
                    z_nabla2_e, area_edge, vn_in, fac_bdydiff_v
                ),
            )
        else:
            vn_new = np.where(
                condition,
                apply_nabla2_and_nabla4_global_to_vn_numpy(
                    area_edge=area_edge,
                    kh_smag_e=kh_smag_limited,
                    z_nabla2_e=z_nabla2_e,
                    z_nabla4_e2=z_nabla4_e2,
                    diff_multfac_vn=diff_multfac_vn,
                    vn=vn_in,
                ),
                vn_in,
            )
        out_vn = vn.copy()
        vn_slice = (
            slice(edge_horizontal_start, edge_horizontal_end),
            slice(vertical_start, vertical_end),
        )
        out_vn[vn_slice] = vn_new[vn_slice]

        # Diffusion of w and horizontal gradients for turbulence
        out_w = w_old.copy()
        out_dwdx = dwdx.copy()
        out_dwdy = dwdy.copy()
        if compute_hgrad_w:
            grad_dwdx, grad_dwdy = calculate_horizontal_gradients_for_turbulence_numpy(
                connectivities, w_old, geofac_grg_x, geofac_grg_y
            )
            grad_slice = (
                slice(hgrad_w_horizontal_start, hgrad_w_horizontal_end),
                slice(vertical_start + 1, vertical_end),
            )
            out_dwdx[grad_slice] = grad_dwdx[grad_slice]
            out_dwdy[grad_slice] = grad_dwdy[grad_slice]

        z_nabla2_c = calculate_nabla2_for_w_numpy(connectivities, w_old, geofac_n2s)
        w_new = apply_nabla2_to_w_numpy(
            connectivities=connectivities,
            area=area,
            z_nabla2_c=z_nabla2_c,
            geofac_n2s=geofac_n2s,
            w=w_old,
            diff_multfac_w=diff_multfac_w,
        )
        w_new = np.where(
            (reshaped_k > 0) & (reshaped_k < nrdmax),
            apply_nabla2_to_w_in_upper_damping_layer_numpy(
                w_new, diff_multfac_n2w, area, z_nabla2_c
            ),
            w_new,
        )
        w_slice = (slice(w_horizontal_start, w_horizontal_end), slice(vertical_start, vertical_end))
        out_w[w_slice] = w_new[w_slice]

        # Diffusion of theta_v and exner with the cold-pool enhancement inlined
        out_theta_v = theta_v.copy()
        out_exner = exner.copy()
        if apply_to_temperature:
            tdiff = (
                theta_v_in
                - np.sum(np.where((c2e2c != -1)[:, :, np.newaxis], theta_v_in[c2e2c], 0), axis=1)
                / 3
            )
            trefdiff = (
                theta_ref_mc
                - np.sum(np.where((c2e2c != -1)[:, :, np.newaxis], theta_ref_mc[c2e2c], 0), axis=1)
                / 3
            )
            enh_diffu_3d = np.where(
                ((tdiff - trefdiff) < thresh_tdiff) & (trefdiff < 0)
                | (tdiff - trefdiff < 1.5 * thresh_tdiff),
                (thresh_tdiff - tdiff + trefdiff) * 5.0e-4,
                smallest_vpfloat,
            )
            k_full = np.arange(theta_v_in.shape[1])
            kh_smag_enhanced = np.where(
                k_full[np.newaxis, :] >= nlev - 2,
                np.maximum(
                    kh_smag_limited,
                    np.max(
                        np.where((e2c != -1)[:, :, np.newaxis], enh_diffu_3d[e2c], -math.inf),
                        axis=1,
                    ),
                ),
                kh_smag_limited,
            )
            z_nabla2_e_theta = (
                kh_smag_enhanced
                * inv_dual_edge_length[:, np.newaxis]
                * (theta_v_in[e2c[:, 1]] - theta_v_in[e2c[:, 0]])
            )
            z_temp = np.sum(z_nabla2_e_theta[c2e] * geofac_div[:, :, np.newaxis], axis=1)
            if apply_zdiffusion_t:
                z_temp = np.where(
                    zd_diffcoef != 0.0,
                    truly_horizontal_diffusion_nabla_of_theta_over_steep_points_numpy(
                        connectivities=connectivities,
                        zd_vertoffset=zd_vertoffset,
                        zd_diffcoef=zd_diffcoef,
                        geofac_n2s_c=geofac_n2s_c,
                        geofac_n2s_nbh=geofac_n2s_nbh,
                        vcoef=vcoef,
                        theta_v=theta_v_in,
                        z_temp=z_temp,
                    ),
                    z_temp,
                )
            theta_v_new, exner_new = update_theta_and_exner_numpy(
                z_temp, area, theta_v_in, exner, rd_o_cvd
            )
            theta_slice = (
                slice(cell_horizontal_start, cell_horizontal_end),
                slice(vertical_start, vertical_end),
            )
            out_theta_v[theta_slice] = theta_v_new[theta_slice]
            out_exner[theta_slice] = exner_new[theta_slice]

        return dict(
            vn=out_vn,
            w=out_w,
            dwdx=out_dwdx,
            dwdy=out_dwdy,
            theta_v=out_theta_v,
            exner=out_exner,
            div_ic=out_div_ic,
            hdef_ic=out_hdef_ic,
        )

    @stencil_tests.input_data_fixture(
        params=[
            {
                "compute_flow_deformation": cfd,
                "compute_hgrad_w": chw,
                "apply_to_temperature": att,
            }
            for cfd, chw, att in itertools.product([True, False], repeat=3)
            # the horizontal w-gradients are only computed as part of the flow-deformation block
            # exclude the case (compute_hgrad_w, compute_flow_deformation) == (True, False)
            if cfd or not chw
        ],
        ids=lambda param: (
            f"compute_flow_deformation[{param['compute_flow_deformation']}]"
            f"-compute_hgrad_w[{param['compute_hgrad_w']}]"
            f"-apply_to_temperature[{param['apply_to_temperature']}]"
        ),
    )
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper,
        grid: base.Grid,
        request: pytest.FixtureRequest,
    ) -> dict:
        compute_flow_deformation = request.param["compute_flow_deformation"]
        compute_hgrad_w = request.param["compute_hgrad_w"]
        apply_to_temperature = request.param["apply_to_temperature"]

        edge_domain = h_grid.domain(dims.EdgeDim)
        cell_domain = h_grid.domain(dims.CellDim)
        limited_area = grid.limited_area if hasattr(grid, "limited_area") else True
        nlev = grid.num_levels

        # vn diffusion
        u_vert = data_alloc.random_field(dims.VertexDim, dims.KDim, dtype=ta.vpfloat)
        v_vert = data_alloc.random_field(dims.VertexDim, dims.KDim, dtype=ta.vpfloat)
        primal_normal_vert_x = data_alloc.random_field(
            dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        primal_normal_vert_y = data_alloc.random_field(
            dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        inv_vert_vert_length = data_alloc.random_field(dims.EdgeDim, dtype=ta.wpfloat)
        inv_primal_edge_length = data_alloc.random_field(dims.EdgeDim, dtype=ta.wpfloat)
        area_edge = data_alloc.random_field(dims.EdgeDim, dtype=ta.wpfloat)
        diff_multfac_vn = data_alloc.random_field(dims.KDim, dtype=ta.wpfloat)
        nudgecoeff_e = data_alloc.random_field(dims.EdgeDim, dtype=ta.wpfloat)
        nudgezone_diff = ta.vpfloat(9.0)
        fac_bdydiff_v = ta.wpfloat(5.0)
        vn_in = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        vn = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        z_nabla2_e = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        # Smagorinsky coefficient
        kh_smag = data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        smag_limit = data_alloc.random_field(dims.KDim, dtype=ta.vpfloat)
        smag_offset = ta.vpfloat("9.0")
        diff_multfac_smag = data_alloc.random_field(dims.KDim, dtype=ta.vpfloat)

        # turbulence diagnostics
        e_bln_c_s = data_alloc.random_field(dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)
        geofac_div = data_alloc.random_field(dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)
        wgtfac_c = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)
        div_ic = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)
        hdef_ic = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)

        # w diffusion
        area = data_alloc.random_field(dims.CellDim, dtype=ta.wpfloat)
        geofac_n2s = data_alloc.random_field(dims.CellDim, dims.C2E2CODim, dtype=ta.wpfloat)
        geofac_grg_x = data_alloc.random_field(dims.CellDim, dims.C2E2CODim, dtype=ta.wpfloat)
        geofac_grg_y = data_alloc.random_field(dims.CellDim, dims.C2E2CODim, dtype=ta.wpfloat)
        w_old = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat)
        w = data_alloc.as_field(w_old.asnumpy().copy(), dims.CellDim, dims.KHalfDim)
        dwdx = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)
        dwdy = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=ta.vpfloat)
        diff_multfac_w = ta.wpfloat(5.0)
        diff_multfac_n2w = data_alloc.random_field(dims.KHalfDim, dtype=ta.wpfloat)
        nrdmax = 13

        # theta_v and exner diffusion
        theta_v_in = data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        theta_v = data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        exner = data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        inv_dual_edge_length = data_alloc.random_field(dims.EdgeDim, dtype=ta.wpfloat)
        zd_vertoffset_buffer = np.zeros(
            (grid.size[dims.CellDim], grid.size[dims.C2E2CDim], grid.size[dims.KDim]),
            dtype=gtx.int32,
        )
        rng = np.random.default_rng()
        for k in range(grid.num_levels):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            zd_vertoffset_buffer[:, :, k] = rng.integers(
                low=0 - k,
                high=grid.num_levels - k - 1,
                size=zd_vertoffset_buffer.shape[:2],
            )
        zd_vertoffset = data_alloc.as_field(
            zd_vertoffset_buffer, dims.CellDim, dims.C2E2CDim, dims.KDim
        )
        zd_diffcoef = data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        geofac_n2s_c = data_alloc.random_field(dims.CellDim, dtype=ta.wpfloat)
        geofac_n2s_nbh = data_alloc.random_field(dims.CellDim, dims.C2E2CDim, dtype=ta.wpfloat)
        vcoef = data_alloc.random_field(dims.CellDim, dims.C2E2CDim, dims.KDim, dtype=ta.wpfloat)
        theta_ref_mc = data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        thresh_tdiff = ta.wpfloat("5.0")
        smallest_vpfloat = constants.VP_NEG_HUGE
        rd_o_cvd = ta.vpfloat("5.0")
        apply_zdiffusion_t = True

        return dict(
            kh_smag=kh_smag,
            smag_limit=smag_limit,
            smag_offset=smag_offset,
            vn_in=vn_in,
            e_bln_c_s=e_bln_c_s,
            geofac_div=geofac_div,
            diff_multfac_smag=diff_multfac_smag,
            wgtfac_c=wgtfac_c,
            div_ic=div_ic,
            hdef_ic=hdef_ic,
            u_vert=u_vert,
            v_vert=v_vert,
            primal_normal_vert_x=primal_normal_vert_x,
            primal_normal_vert_y=primal_normal_vert_y,
            z_nabla2_e=z_nabla2_e,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_primal_edge_length=inv_primal_edge_length,
            area_edge=area_edge,
            diff_multfac_vn=diff_multfac_vn,
            nudgecoeff_e=nudgecoeff_e,
            nudgezone_diff=nudgezone_diff,
            fac_bdydiff_v=fac_bdydiff_v,
            area=area,
            geofac_n2s=geofac_n2s,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            w_old=w_old,
            compute_hgrad_w=compute_hgrad_w,
            dwdx=dwdx,
            dwdy=dwdy,
            diff_multfac_w=diff_multfac_w,
            diff_multfac_n2w=diff_multfac_n2w,
            theta_v_in=theta_v_in,
            inv_dual_edge_length=inv_dual_edge_length,
            zd_vertoffset=zd_vertoffset,
            zd_diffcoef=zd_diffcoef,
            geofac_n2s_c=geofac_n2s_c,
            geofac_n2s_nbh=geofac_n2s_nbh,
            vcoef=vcoef,
            theta_v=theta_v,
            exner=exner,
            theta_ref_mc=theta_ref_mc,
            thresh_tdiff=thresh_tdiff,
            smallest_vpfloat=smallest_vpfloat,
            rd_o_cvd=rd_o_cvd,
            apply_zdiffusion_t=apply_zdiffusion_t,
            compute_flow_deformation=compute_flow_deformation,
            apply_to_temperature=apply_to_temperature,
            limited_area=limited_area,
            vn=vn,
            w=w,
            start_2nd_nudge_line_idx_e=grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2)),
            nrdmax=nrdmax,
            nlev=nlev,
            w_horizontal_start=grid.start_index(cell_domain(h_grid.Zone.INTERIOR)),
            w_horizontal_end=grid.end_index(cell_domain(h_grid.Zone.LOCAL)),
            edge_horizontal_start=grid.start_index(
                edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
            ),
            edge_horizontal_end=grid.end_index(edge_domain(h_grid.Zone.LOCAL)),
            hgrad_w_horizontal_start=(
                grid.start_index(cell_domain(h_grid.Zone.NUDGING))
                if grid.limited_area
                else grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
            ),
            hgrad_w_horizontal_end=grid.end_index(cell_domain(h_grid.Zone.HALO)),
            cell_horizontal_start=grid.start_index(cell_domain(h_grid.Zone.NUDGING)),
            cell_horizontal_end=grid.end_index(cell_domain(h_grid.Zone.LOCAL)),
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
