# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import math

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.diffusion_for_prognostic_fields import (
    diffusion_for_prognostic_fields,
)
from icon4py.model.common import constants, dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests

from .test_apply_nabla2_and_nabla4_global_to_vn import apply_nabla2_and_nabla4_global_to_vn_numpy
from .test_apply_nabla2_and_nabla4_to_vn import apply_nabla2_and_nabla4_to_vn_numpy
from .test_apply_nabla2_to_vn_in_lateral_boundary import (
    apply_nabla2_to_vn_in_lateral_boundary_numpy,
)
from .test_apply_nabla2_to_w import apply_nabla2_to_w_numpy
from .test_apply_nabla2_to_w_in_upper_damping_layer import (
    apply_nabla2_to_w_in_upper_damping_layer_numpy,
)
from .test_calculate_diagnostics_for_turbulence import calculate_diagnostics_for_turbulence_numpy
from .test_calculate_horizontal_gradients_for_turbulence import (
    calculate_horizontal_gradients_for_turbulence_numpy,
)
from .test_calculate_nabla2_for_w import calculate_nabla2_for_w_numpy
from .test_calculate_nabla2_for_z import calculate_nabla2_for_z_numpy
from .test_calculate_nabla2_of_theta import calculate_nabla2_of_theta_numpy
from .test_calculate_nabla4 import calculate_nabla4_numpy
from .test_truly_horizontal_diffusion_nabla_of_theta_over_steep_points import (
    truly_horizontal_diffusion_nabla_of_theta_over_steep_points_numpy,
)
from .test_update_theta_and_exner import update_theta_and_exner_numpy


@pytest.mark.uses_concat_where
@pytest.mark.continuous_benchmarking
class TestDiffusionForPrognosticFields(stencil_tests.StencilTest):
    PROGRAM = diffusion_for_prognostic_fields
    OUTPUTS = ("vn", "w", "dwdx", "dwdy", "theta_v", "exner", "div_ic", "hdef_ic")
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "compute_diagnostic_quantities",
            "apply_to_temperature",
            "limited_area",
            "type_shear",
            "start_2nd_nudge_line_idx_e",
            "nrdmax",
            "nlev",
            "interior_idx",
            "halo_idx",
            "edge_horizontal_start",
            "edge_horizontal_end",
            "w_horizontal_start",
            "w_horizontal_end",
            "cell_horizontal_start",
            "cell_horizontal_end",
            "vertical_start",
            "vertical_end",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "compute_diagnostic_quantities",
            "apply_to_temperature",
            "type_shear",
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
        primal_normal_vert_v1: np.ndarray,
        primal_normal_vert_v2: np.ndarray,
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
        type_shear: int,
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
        compute_diagnostic_quantities: bool,
        apply_to_temperature: bool,
        limited_area: bool,
        start_2nd_nudge_line_idx_e: int,
        nrdmax: int,
        nlev: int,
        interior_idx: int,
        halo_idx: int,
        edge_horizontal_start: int,
        edge_horizontal_end: int,
        w_horizontal_start: int,
        w_horizontal_end: int,
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
        reshaped_cell = np.arange(w_old.shape[0])[:, np.newaxis]

        # Smagorinsky limiter
        kh_smag_limited = np.maximum(0.0, kh_smag - smag_offset)
        kh_smag_limited = np.minimum(kh_smag_limited, smag_limit)

        # Turbulence diagnostics from pre-diffusion vn
        out_div_ic = div_ic.copy()
        out_hdef_ic = hdef_ic.copy()
        if compute_diagnostic_quantities:
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
            primal_normal_vert_v1=primal_normal_vert_v1,
            primal_normal_vert_v2=primal_normal_vert_v2,
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
        if type_shear == 2:
            grad_dwdx, grad_dwdy = calculate_horizontal_gradients_for_turbulence_numpy(
                connectivities, w_old, geofac_grg_x, geofac_grg_y
            )
            grad_slice = (
                slice(w_horizontal_start, w_horizontal_end),
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
            (reshaped_k > 0)
            & (reshaped_k < nrdmax)
            & (interior_idx <= reshaped_cell)
            & (reshaped_cell < halo_idx),
            apply_nabla2_to_w_in_upper_damping_layer_numpy(
                w_new, diff_multfac_n2w, area, z_nabla2_c
            ),
            w_new,
        )
        w_slice = (slice(interior_idx, halo_idx), slice(vertical_start, vertical_end))
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
            z_nabla2_e_theta = calculate_nabla2_for_z_numpy(
                connectivities,
                kh_smag_enhanced,
                inv_dual_edge_length,
                theta_v_in,
                np.zeros_like(kh_smag_enhanced),
            )
            z_temp = calculate_nabla2_of_theta_numpy(connectivities, z_nabla2_e_theta, geofac_div)
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
            {"compute_diagnostic_quantities": flag[0], "apply_to_temperature": flag[1]}
            for flag in [(True, True), (False, True), (True, False), (False, False)]
        ],
        ids=lambda param: (
            f"compute_diagnostic_quantities[{param['compute_diagnostic_quantities']}]"
            f"-apply_to_temperature[{param['apply_to_temperature']}]"
        ),
    )
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper,
        grid: base.Grid,
        request: pytest.FixtureRequest,
    ) -> dict:
        compute_diagnostic_quantities = request.param["compute_diagnostic_quantities"]
        apply_to_temperature = request.param["apply_to_temperature"]

        edge_domain = h_grid.domain(dims.EdgeDim)
        cell_domain = h_grid.domain(dims.CellDim)
        limited_area = grid.limited_area if hasattr(grid, "limited_area") else True
        nlev = grid.num_levels

        def _get_start_index_for_w_diffusion() -> gtx.int32:
            return (
                grid.start_index(cell_domain(h_grid.Zone.NUDGING))
                if grid.limited_area
                else grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
            )

        # vn diffusion
        u_vert = data_alloc.random_field(dims.VertexDim, dims.KDim, dtype=ta.vpfloat)
        v_vert = data_alloc.random_field(dims.VertexDim, dims.KDim, dtype=ta.vpfloat)
        primal_normal_vert_v1 = data_alloc.random_field(
            dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        primal_normal_vert_v2 = data_alloc.random_field(
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
        type_shear = 2
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
            primal_normal_vert_v1=primal_normal_vert_v1,
            primal_normal_vert_v2=primal_normal_vert_v2,
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
            type_shear=type_shear,
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
            compute_diagnostic_quantities=compute_diagnostic_quantities,
            apply_to_temperature=apply_to_temperature,
            limited_area=limited_area,
            vn=vn,
            w=w,
            start_2nd_nudge_line_idx_e=grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2)),
            nrdmax=nrdmax,
            nlev=nlev,
            interior_idx=grid.start_index(cell_domain(h_grid.Zone.INTERIOR)),
            halo_idx=grid.end_index(cell_domain(h_grid.Zone.LOCAL)),
            edge_horizontal_start=grid.start_index(
                edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
            ),
            edge_horizontal_end=grid.end_index(edge_domain(h_grid.Zone.LOCAL)),
            w_horizontal_start=_get_start_index_for_w_diffusion(),
            w_horizontal_end=grid.end_index(cell_domain(h_grid.Zone.HALO)),
            cell_horizontal_start=grid.start_index(cell_domain(h_grid.Zone.NUDGING)),
            cell_horizontal_end=grid.end_index(cell_domain(h_grid.Zone.LOCAL)),
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
