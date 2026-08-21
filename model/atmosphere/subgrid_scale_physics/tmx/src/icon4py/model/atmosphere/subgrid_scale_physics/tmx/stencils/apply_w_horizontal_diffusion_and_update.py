# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, C2EDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _apply_w_horizontal_diffusion_and_update(
    hori_tend_e: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    inv_rho_ic: fa.CellKField[wpfloat],
    w: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Add the horizontal w-diffusion tendency (interpolated to cell centers) to
    the w tendency and update w.

    Port of the last two loops of 'Compute_diffusion_vert_wind' (mo_vdf.f90):

        tend(k) += inv_rho_ic(k) * sum_{e in C2E} e_bln_c_s * hori_tend_e(e, k)
        new_w(k) = w(k) + tend(k) * dtime

    All K fields live on half levels (nlev + 1 rows). The top and bottom half
    levels are excluded (w = 0 boundary condition): rows jk = 2..nlev
    (1-based), i.e. the program must be called with vertical bounds (1, nlev).
    tend is read and written in place ('out=(new_w, tend)').

    Domains (Fortran caller, the tmx 'domain' cell loop bounds): cells from
    rl_start = grf_bdywidth_c + 1 -> 'h_grid.Zone.NUDGING' to
    rl_end = min_rlcell_int -> 'h_grid.Zone.LOCAL'.
    """
    new_tend = tend + inv_rho_ic * neighbor_sum(e_bln_c_s * hori_tend_e(C2E), axis=C2EDim)
    new_w = w + new_tend * dtime
    return new_w, new_tend


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_w_horizontal_diffusion_and_update(
    hori_tend_e: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    inv_rho_ic: fa.CellKField[wpfloat],
    w: fa.CellKField[wpfloat],
    new_w: fa.CellKField[wpfloat],
    tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_w_horizontal_diffusion_and_update(
        hori_tend_e=hori_tend_e,
        e_bln_c_s=e_bln_c_s,
        inv_rho_ic=inv_rho_ic,
        w=w,
        tend=tend,
        dtime=dtime,
        out=(new_w, tend),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
