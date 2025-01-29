# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.experimental import as_offset
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2EC, Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_hydrostatic_correction_term(
    theta_v: fa.CellKField[wpfloat],
    ikoffset: gtx.Field[gtx.Dims[dims.ECDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[gtx.Dims[dims.ECDim, dims.KDim], vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    grav_o_cpd: wpfloat,
) -> fa.EdgeKField[vpfloat]:
    """
    Formerly known as _mo_solve_nonhydro_stencil_21.

    # scidoc:
    # Outputs:
    #  - z_hydro_corr :
    #     $$
    #     \exnhydrocorr{\e} = \frac{g}{\cpd} \Wedge 4 \frac{ \vpotemp{}{\c_1}{\k} - \vpotemp{}{\c_0}{\k} }{ (\vpotemp{}{\c_1}{\k} + \vpotemp{}{\c_0}{\k})^2 },
    #     $$
    #     with
    #     $$
    #     \vpotemp{}{\c_i}{\k} = \vpotemp{}{\c_i}{\k^*} + \dzgradp \frac{\vpotemp{}{\c_i}{\k^*-1/2} - \vpotemp{}{\c_i}{\k^*+1/2}}{\Dz{\k^*}}
    #     $$
    #     Compute the hydrostatically approximated correction term that
    #     replaces the downward extrapolation (last term in eq. 10 in
    #     |ICONSteepSlopePressurePaper|).
    #     This is only computed for the bottom-most level because all
    #     edges which have a neighboring cell center inside terrain
    #     beyond a certain limit use the same correction term at $k^*$
    #     level in eq. 10 in |ICONSteepSlopePressurePaper| (see also the
    #     last paragraph on page 3724 for the discussion).
    #     $\c_i$ are the indexes of the adjacent cell centers using
    #     $\offProv{e2c}$;
    #     $k^*$ is the level index of the neighboring (horizontally, not
    #     terrain-following) cell center and $h^*$ is its height.
    #
    # Inputs:
    #  - $\vpotemp{}{\c}{\k}$ : theta_v
    #  - $\vpotemp{}{\c}{\k\pm1/2}$ : theta_v_ic
    #  - $\frac{g}{\cpd}$ : grav_o_cpd
    #  - $\Wedge$ : inverse_dual_edge_lengths
    #  - $1 / \Dz{\k}$ : inv_ddqz_z_full
    #  - $\dzgradp$ : zdiff_gradp
    #  - $\k^*$ : vertoffset_gradp
    #

    """
    zdiff_gradp_wp = astype(zdiff_gradp, wpfloat)

    theta_v_0 = theta_v(E2C[0])(as_offset(Koff, ikoffset(E2EC[0])))
    theta_v_1 = theta_v(E2C[1])(as_offset(Koff, ikoffset(E2EC[1])))

    theta_v_ic_0 = theta_v_ic(E2C[0])(as_offset(Koff, ikoffset(E2EC[0])))
    theta_v_ic_1 = theta_v_ic(E2C[1])(as_offset(Koff, ikoffset(E2EC[1])))

    theta_v_ic_p1_0 = theta_v_ic(E2C[0])(as_offset(Koff, ikoffset(E2EC[0]) + 1))
    theta_v_ic_p1_1 = theta_v_ic(E2C[1])(as_offset(Koff, ikoffset(E2EC[1]) + 1))

    inv_ddqz_z_full_0_wp = astype(
        inv_ddqz_z_full(E2C[0])(as_offset(Koff, ikoffset(E2EC[0]))), wpfloat
    )
    inv_ddqz_z_full_1_wp = astype(
        inv_ddqz_z_full(E2C[1])(as_offset(Koff, ikoffset(E2EC[1]))), wpfloat
    )

    z_theta_0 = (
        theta_v_0
        + zdiff_gradp_wp(E2EC[0]) * (theta_v_ic_0 - theta_v_ic_p1_0) * inv_ddqz_z_full_0_wp
    )
    z_theta_1 = (
        theta_v_1
        + zdiff_gradp_wp(E2EC[1]) * (theta_v_ic_1 - theta_v_ic_p1_1) * inv_ddqz_z_full_1_wp
    )
    z_hydro_corr_wp = (
        grav_o_cpd
        * inv_dual_edge_length
        * (z_theta_1 - z_theta_0)
        * wpfloat("4.0")
        / ((z_theta_0 + z_theta_1) * (z_theta_0 + z_theta_1))
    )

    return astype(z_hydro_corr_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def compute_hydrostatic_correction_term(
    theta_v: fa.CellKField[wpfloat],
    ikoffset: gtx.Field[gtx.Dims[dims.ECDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[gtx.Dims[dims.ECDim, dims.KDim], vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    grav_o_cpd: wpfloat,
    z_hydro_corr: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_hydrostatic_correction_term(
        theta_v,
        ikoffset,
        zdiff_gradp,
        theta_v_ic,
        inv_ddqz_z_full,
        inv_dual_edge_length,
        grav_o_cpd,
        out=z_hydro_corr,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
