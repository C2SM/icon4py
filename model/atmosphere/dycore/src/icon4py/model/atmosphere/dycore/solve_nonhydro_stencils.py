# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.dycore.dycore_utils import (
    _broadcast_zero_to_three_edge_kdim_fields_wp,
)
from icon4py.model.atmosphere.dycore.stencils.compute_perturbation_of_rho_and_theta import (
    _compute_perturbation_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers import (
    _compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers,
)
from icon4py.model.atmosphere.dycore.stencils.compute_virtual_potential_temperatures_and_pressure_gradient import (
    _compute_virtual_potential_temperatures_and_pressure_gradient,
)
from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_wp import (
    _init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.atmosphere.dycore.stencils.update_density_exner_wind import (
    _update_density_exner_wind,
)
from icon4py.model.atmosphere.dycore.stencils.update_wind import _update_wind
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def init_test_fields(
    z_rho_e: fa.EdgeKField[wpfloat],
    z_theta_v_e: fa.EdgeKField[wpfloat],
    z_dwdz_dd: fa.CellKField[wpfloat],
    z_graddiv_vn: fa.EdgeKField[wpfloat],
    edges_start: gtx.int32,
    edges_end: gtx.int32,
    cells_start: gtx.int32,
    cells_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _broadcast_zero_to_three_edge_kdim_fields_wp(
        out=(z_rho_e, z_theta_v_e, z_graddiv_vn),
        domain={dims.EdgeDim: (edges_start, edges_end), dims.KDim: (vertical_start, vertical_end)},
    )
    _init_cell_kdim_field_with_zero_wp(
        out=z_dwdz_dd,
        domain={dims.CellDim: (cells_start, cells_end), dims.KDim: (vertical_start, vertical_end)},
    )


@gtx.field_operator
def _compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures(
    rho: fa.CellKField[wpfloat],
    z_rth_pr_1: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
    rho_ref_mc: fa.CellKField[vpfloat],
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    rho_ic: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    exner_pr: fa.CellKField[wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[vpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    z_theta_v_pr_ic: fa.CellKField[vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    z_th_ddz_exner_c: fa.CellKField[vpfloat],
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[vpfloat],
]:
    (z_rth_pr_1, z_rth_pr_2) = concat_where(
        dims.KDim == 0,
        _compute_perturbation_of_rho_and_theta(rho, rho_ref_mc, theta_v, theta_ref_mc),
        (z_rth_pr_1, z_rth_pr_2),
    )

    (rho_ic, z_rth_pr_1, z_rth_pr_2) = concat_where(
        dims.KDim >= 1,
        _compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers(
            wgtfac_c, rho, rho_ref_mc, theta_v, theta_ref_mc
        ),
        (rho_ic, z_rth_pr_1, z_rth_pr_2),
    )

    (z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c) = concat_where(
        dims.KDim >= 1,
        _compute_virtual_potential_temperatures_and_pressure_gradient(
            wgtfac_c,
            z_rth_pr_2,
            theta_v,
            vwind_expl_wgt,
            exner_pr,
            d_exner_dz_ref_ic,
            ddqz_z_half,
        ),
        (z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c),
    )

    return z_rth_pr_1, z_rth_pr_2, rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def stencils_61_62(
    rho_now: fa.CellKField[float],
    grf_tend_rho: fa.CellKField[float],
    theta_v_now: fa.CellKField[float],
    grf_tend_thv: fa.CellKField[float],
    w_now: fa.CellKField[float],
    grf_tend_w: fa.CellKField[float],
    rho_new: fa.CellKField[float],
    exner_new: fa.CellKField[float],
    w_new: fa.CellKField[float],
    dtime: float,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_density_exner_wind(
        rho_now,
        grf_tend_rho,
        theta_v_now,
        grf_tend_thv,
        w_now,
        grf_tend_w,
        dtime,
        out=(rho_new, exner_new, w_new),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _update_wind(
        w_now,
        grf_tend_w,
        dtime,
        out=w_new,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
