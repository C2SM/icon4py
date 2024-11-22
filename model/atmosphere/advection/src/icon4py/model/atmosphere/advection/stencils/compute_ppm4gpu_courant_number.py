# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import abs, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff


# TODO (dastrm): this stencil has no test
# TODO (dastrm): this stencil does not strictly match the fortran code


@gtx.field_operator
def _compute_ppm4gpu_courant_number(
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slevp1_ti: gtx.int32,
    nlev: gtx.int32,
    dbl_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    z_mass_p = p_dtime * p_mflx_contra_v
    z_mass_pos = z_mass_p > 0.0

    # no while loop iterations
    p_cellmass_now_jks = p_cellmass_now

    # one while loop iteration
    in_bounds = k <= nlev - 1
    mass_gt_cellmass = where(z_mass_pos & in_bounds, z_mass_p >= p_cellmass_now, False)
    z_mass_p = z_mass_p - where(mass_gt_cellmass, p_cellmass_now, 0.0)
    z_cfl = z_cfl + where(mass_gt_cellmass, 1.0, 0.0)
    p_cellmass_now_jks = where(mass_gt_cellmass, p_cellmass_now(Koff[1]), p_cellmass_now_jks)

    # two while loop iterations
    in_bounds = k <= nlev - 2
    mass_gt_cellmass = mass_gt_cellmass & where(
        z_mass_pos & in_bounds, z_mass_p >= p_cellmass_now(Koff[1]), False
    )
    z_mass_p = z_mass_p - where(mass_gt_cellmass, p_cellmass_now(Koff[1]), 0.0)
    z_cfl = z_cfl + where(mass_gt_cellmass, 1.0, 0.0)
    p_cellmass_now_jks = where(mass_gt_cellmass, p_cellmass_now(Koff[2]), p_cellmass_now_jks)

    # three while loop iterations
    in_bounds = k <= nlev - 3
    mass_gt_cellmass = mass_gt_cellmass & where(
        z_mass_pos & in_bounds, z_mass_p >= p_cellmass_now(Koff[2]), False
    )
    z_mass_p = z_mass_p - where(mass_gt_cellmass, p_cellmass_now(Koff[2]), 0.0)
    z_cfl = z_cfl + where(mass_gt_cellmass, 1.0, 0.0)
    p_cellmass_now_jks = where(mass_gt_cellmass, p_cellmass_now(Koff[3]), p_cellmass_now_jks)

    # four while loop iterations
    in_bounds = k <= nlev - 4
    mass_gt_cellmass = mass_gt_cellmass & where(
        z_mass_pos & in_bounds, z_mass_p >= p_cellmass_now(Koff[3]), False
    )
    z_mass_p = z_mass_p - where(mass_gt_cellmass, p_cellmass_now(Koff[3]), 0.0)
    z_cfl = z_cfl + where(mass_gt_cellmass, 1.0, 0.0)
    p_cellmass_now_jks = where(mass_gt_cellmass, p_cellmass_now(Koff[4]), p_cellmass_now_jks)

    z_cflfrac = where(z_mass_pos, z_mass_p / p_cellmass_now_jks, 0.0)
    z_cfl = where(z_cflfrac < 1.0, z_cfl + z_cflfrac, z_cfl + 1.0 - dbl_eps)

    z_mass_n = p_dtime * p_mflx_contra_v
    z_mass_neg = z_mass_n <= 0.0

    # no while loop iterations
    p_cellmass_now_jks = p_cellmass_now(Koff[-1])

    # one while loop iteration
    in_bounds = k >= slevp1_ti + 1
    mass_gt_cellmass = where(
        z_mass_neg & in_bounds, abs(z_mass_n) >= p_cellmass_now(Koff[-1]), False
    )
    z_mass_n = z_mass_n + where(mass_gt_cellmass, p_cellmass_now(Koff[-1]), 0.0)
    z_cfl = z_cfl - where(mass_gt_cellmass, 1.0, 0.0)
    p_cellmass_now_jks = where(mass_gt_cellmass, p_cellmass_now(Koff[-2]), p_cellmass_now_jks)

    # two while loop iterations
    in_bounds = k >= slevp1_ti + 2
    mass_gt_cellmass = mass_gt_cellmass & where(
        z_mass_neg & in_bounds, abs(z_mass_n) >= p_cellmass_now(Koff[-2]), False
    )
    z_mass_n = z_mass_n + where(mass_gt_cellmass, p_cellmass_now(Koff[-2]), 0.0)
    z_cfl = z_cfl - where(mass_gt_cellmass, 1.0, 0.0)
    p_cellmass_now_jks = where(mass_gt_cellmass, p_cellmass_now(Koff[-3]), p_cellmass_now_jks)

    # three while loop iterations
    in_bounds = k >= slevp1_ti + 3
    mass_gt_cellmass = mass_gt_cellmass & where(
        z_mass_neg & in_bounds, abs(z_mass_n) >= p_cellmass_now(Koff[-3]), False
    )
    z_mass_n = z_mass_n + where(mass_gt_cellmass, p_cellmass_now(Koff[-3]), 0.0)
    z_cfl = z_cfl - where(mass_gt_cellmass, 1.0, 0.0)
    p_cellmass_now_jks = where(mass_gt_cellmass, p_cellmass_now(Koff[-4]), p_cellmass_now_jks)

    # four while loop iterations
    in_bounds = k >= slevp1_ti + 4
    mass_gt_cellmass = mass_gt_cellmass & where(
        z_mass_neg & in_bounds, abs(z_mass_n) >= p_cellmass_now(Koff[-4]), False
    )
    z_mass_n = z_mass_n + where(mass_gt_cellmass, p_cellmass_now(Koff[-4]), 0.0)
    z_cfl = z_cfl - where(mass_gt_cellmass, 1.0, 0.0)
    p_cellmass_now_jks = where(mass_gt_cellmass, p_cellmass_now(Koff[-5]), p_cellmass_now_jks)

    z_cflfrac = where(z_mass_neg, z_mass_n / p_cellmass_now_jks, 0.0)
    z_cfl = where(abs(z_cflfrac) < 1.0, z_cfl + z_cflfrac, z_cfl + dbl_eps - 1.0)

    return z_cfl


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm4gpu_courant_number(
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slevp1_ti: gtx.int32,
    nlev: gtx.int32,
    dbl_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_ppm4gpu_courant_number(
        p_mflx_contra_v,
        p_cellmass_now,
        z_cfl,
        k,
        slevp1_ti,
        nlev,
        dbl_eps,
        p_dtime,
        out=z_cfl,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
