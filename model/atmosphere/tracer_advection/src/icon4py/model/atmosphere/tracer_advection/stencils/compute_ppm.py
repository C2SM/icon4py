# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Stencils of the piecewise parabolic method (PPM) for vertical tracer advection."""

import gt4py.next as gtx
from gt4py.next import abs, astype, broadcast, floor, where  # noqa: A004
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.tracer_advection.stencils.compute_vertical_parabola_limiter_condition import (
    _compute_vertical_parabola_limiter_condition,
)
from icon4py.model.atmosphere.tracer_advection.stencils.limit_vertical_parabola_semi_monotonically import (
    _limit_vertical_parabola_semi_monotonically,
)
from icon4py.model.atmosphere.tracer_advection.stencils.limit_vertical_slope_semi_monotonically import (
    _limit_vertical_slope_semi_monotonically,
)
from icon4py.model.atmosphere.tracer_advection.tracer_advection_type import VerticalAdvectionLimiter
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_ppm_quadratic_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
) -> fa.CellKHalfField[ta.wpfloat]:
    hgt = p_cellhgt_mc_now(dims.KHalfDim + 0.5)
    hgt_m1 = p_cellhgt_mc_now(dims.KHalfDim - 0.5)
    cc = p_cc(dims.KHalfDim + 0.5)
    cc_m1 = p_cc(dims.KHalfDim - 0.5)
    p_face = cc * (wpfloat(1.0) - (hgt / hgt_m1)) + (hgt / (hgt_m1 + hgt)) * (
        (hgt / hgt_m1) * cc + cc_m1
    )

    return p_face


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_quadratic_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    p_face: fa.CellKHalfField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_quadratic_face_values(
        p_cc=p_cc,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        out=p_face,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_ppm_quartic_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
) -> fa.CellKHalfField[ta.wpfloat]:
    hgt_m2 = p_cellhgt_mc_now(dims.KHalfDim - 1.5)
    hgt_m1 = p_cellhgt_mc_now(dims.KHalfDim - 0.5)
    hgt = p_cellhgt_mc_now(dims.KHalfDim + 0.5)
    hgt_p1 = p_cellhgt_mc_now(dims.KHalfDim + 1.5)
    cc_m1 = p_cc(dims.KHalfDim - 0.5)
    cc = p_cc(dims.KHalfDim + 0.5)
    slope_above = z_slope(dims.KHalfDim - 0.5)
    slope_below = z_slope(dims.KHalfDim + 0.5)

    zgeo1 = hgt_m1 / (hgt_m1 + hgt)
    zgeo2 = wpfloat(1.0) / (hgt_m2 + hgt_m1 + hgt + hgt_p1)
    zgeo3 = (hgt_m2 + hgt_m1) / (wpfloat(2.0) * hgt_m1 + hgt)
    zgeo4 = (hgt_p1 + hgt) / (wpfloat(2.0) * hgt + hgt_m1)

    p_face = (
        cc_m1
        + zgeo1 * (cc - cc_m1)
        + zgeo2
        * (
            (wpfloat(2.0) * hgt * zgeo1) * (zgeo3 - zgeo4) * (cc - cc_m1)
            - zgeo3 * hgt_m1 * slope_below
            + zgeo4 * hgt * slope_above
        )
    )

    return p_face


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_quartic_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
    p_face: fa.CellKHalfField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_quartic_face_values(
        p_cc=p_cc,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        z_slope=z_slope,
        out=p_face,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )


# TODO(nfarabullini, OngChia): this stencil is imported but never called
# TODO(nfarabullini, OngChia): slev/elev and vertical_start/end are redundant


@gtx.field_operator
def _compute_ppm_all_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    p_face_in: fa.CellKHalfField[ta.wpfloat],
    slev: gtx.int32,
    elev: gtx.int32,
    slevp1: gtx.int32,
    elevp1: gtx.int32,
) -> fa.CellKHalfField[ta.wpfloat]:
    p_face = concat_where(
        (dims.KHalfDim == slevp1) | (dims.KHalfDim == elev),
        _compute_ppm_quadratic_face_values(p_cc, p_cellhgt_mc_now),
        p_face_in,
    )

    p_face = concat_where(dims.KHalfDim == slev, p_cc(dims.KHalfDim + 0.5), p_face)

    p_face = concat_where(dims.KHalfDim == elevp1, p_cc(dims.KHalfDim - 0.5), p_face)

    return p_face


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_all_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    p_face_in: fa.CellKHalfField[ta.wpfloat],
    p_face: fa.CellKHalfField[ta.wpfloat],
    slev: gtx.int32,
    elev: gtx.int32,
    slevp1: gtx.int32,
    elevp1: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_all_face_values(
        p_cc=p_cc,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        p_face_in=p_face_in,
        slev=slev,
        elev=elev,
        slevp1=slevp1,
        elevp1=elevp1,
        out=p_face,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_ppm_slope_a(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    zfac_m1 = (p_cc - p_cc(dims.KDim - 1)) / (p_cellhgt_mc_now + p_cellhgt_mc_now(dims.KDim - 1))
    zfac = (p_cc(dims.KDim + 1) - p_cc) / (p_cellhgt_mc_now(dims.KDim + 1) + p_cellhgt_mc_now)
    z_slope = (
        p_cellhgt_mc_now
        / (p_cellhgt_mc_now(dims.KDim - 1) + p_cellhgt_mc_now + p_cellhgt_mc_now(dims.KDim + 1))
    ) * (
        (wpfloat(2.0) * p_cellhgt_mc_now(dims.KDim - 1) + p_cellhgt_mc_now) * zfac
        + (p_cellhgt_mc_now + wpfloat(2.0) * p_cellhgt_mc_now(dims.KDim + 1)) * zfac_m1
    )

    return z_slope


@gtx.field_operator
def _compute_ppm_slope_b(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    zfac_m1 = (p_cc - p_cc(dims.KDim - 1)) / (p_cellhgt_mc_now + p_cellhgt_mc_now(dims.KDim - 1))
    z_slope = (
        (p_cellhgt_mc_now / (p_cellhgt_mc_now(dims.KDim - 1) + p_cellhgt_mc_now + p_cellhgt_mc_now))
        * (p_cellhgt_mc_now + wpfloat(2.0) * p_cellhgt_mc_now)
        * zfac_m1
    )

    return z_slope


@gtx.field_operator
def _compute_ppm_slope(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    elev: gtx.int32,
) -> fa.CellKField[ta.wpfloat]:
    z_slope = concat_where(
        dims.KDim == elev,
        _compute_ppm_slope_b(p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now),
        _compute_ppm_slope_a(p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now),
    )

    return z_slope


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_slope(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
    elev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_slope(
        p_cc=p_cc,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        elev=elev,
        out=z_slope,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# TODO(nfarabullini, OngChia): this stencil does not strictly match the fortran code


@gtx.field_operator
def _compute_courant_number_below(
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_mass: fa.CellKHalfField[ta.wpfloat],
    z_cfl: fa.CellKHalfField[ta.wpfloat],
    k_half: fa.KHalfField[gtx.int32],
    nlev: gtx.int32,
    wp_eps: ta.wpfloat,
) -> fa.CellKHalfField[ta.wpfloat]:
    z_mass_pos = z_mass > wpfloat(0.0)

    in_bounds_p0 = k_half <= nlev - 1
    in_bounds_p1 = k_half <= nlev - 2
    in_bounds_p2 = k_half <= nlev - 3
    in_bounds_p3 = k_half <= nlev - 4

    p_cellmass_p0 = concat_where(
        dims.KHalfDim <= nlev - 1, p_cellmass_now(dims.KHalfDim + 0.5), wpfloat(0.0)
    )
    p_cellmass_p1 = concat_where(
        dims.KHalfDim <= nlev - 1, p_cellmass_now(dims.KHalfDim + 1.5), wpfloat(0.0)
    )
    p_cellmass_p2 = concat_where(
        dims.KHalfDim <= nlev - 2, p_cellmass_now(dims.KHalfDim + 2.5), wpfloat(0.0)
    )
    p_cellmass_p3 = concat_where(
        dims.KHalfDim <= nlev - 3, p_cellmass_now(dims.KHalfDim + 3.5), wpfloat(0.0)
    )
    p_cellmass_p4 = concat_where(
        dims.KHalfDim <= nlev - 4, p_cellmass_now(dims.KHalfDim + 4.5), wpfloat(0.0)
    )

    mass_gt_cellmass_p0 = where(z_mass_pos & in_bounds_p0, z_mass >= p_cellmass_p0, False)
    z_mass = z_mass - where(mass_gt_cellmass_p0, p_cellmass_p0, wpfloat(0.0))

    mass_gt_cellmass_p1 = mass_gt_cellmass_p0 & where(
        z_mass_pos & in_bounds_p1, z_mass >= p_cellmass_p1, False
    )
    z_mass = z_mass - where(mass_gt_cellmass_p1, p_cellmass_p1, wpfloat(0.0))

    mass_gt_cellmass_p2 = mass_gt_cellmass_p1 & where(
        z_mass_pos & in_bounds_p2, z_mass >= p_cellmass_p2, False
    )
    z_mass = z_mass - where(mass_gt_cellmass_p2, p_cellmass_p2, wpfloat(0.0))

    mass_gt_cellmass_p3 = mass_gt_cellmass_p2 & where(
        z_mass_pos & in_bounds_p3, z_mass >= p_cellmass_p3, False
    )
    z_mass = z_mass - where(mass_gt_cellmass_p3, p_cellmass_p3, wpfloat(0.0))

    z_cfl = z_cfl + where(mass_gt_cellmass_p0, wpfloat(1.0), wpfloat(0.0))
    z_cfl = z_cfl + where(mass_gt_cellmass_p1, wpfloat(1.0), wpfloat(0.0))
    z_cfl = z_cfl + where(mass_gt_cellmass_p2, wpfloat(1.0), wpfloat(0.0))
    z_cfl = z_cfl + where(mass_gt_cellmass_p3, wpfloat(1.0), wpfloat(0.0))

    p_cellmass_now_jks = concat_where(
        dims.KHalfDim < nlev + 1, p_cellmass_now(dims.KHalfDim + 0.5), wpfloat(0.0)
    )
    p_cellmass_now_jks = where(mass_gt_cellmass_p0, p_cellmass_p1, p_cellmass_now_jks)
    p_cellmass_now_jks = where(mass_gt_cellmass_p1, p_cellmass_p2, p_cellmass_now_jks)
    p_cellmass_now_jks = where(mass_gt_cellmass_p2, p_cellmass_p3, p_cellmass_now_jks)
    p_cellmass_now_jks = where(mass_gt_cellmass_p3, p_cellmass_p4, p_cellmass_now_jks)

    z_cflfrac = where(z_mass_pos, z_mass / p_cellmass_now_jks, wpfloat(0.0))
    z_cfl = z_cfl + where(z_cflfrac < wpfloat(1.0), z_cflfrac, wpfloat(1.0) - wp_eps)

    return z_cfl


@gtx.field_operator
def _compute_courant_number_above(
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_mass: fa.CellKHalfField[ta.wpfloat],
    z_cfl: fa.CellKHalfField[ta.wpfloat],
    k_half: fa.KHalfField[gtx.int32],
    slevp1_ti: gtx.int32,
    wp_eps: ta.wpfloat,
) -> fa.CellKHalfField[ta.wpfloat]:
    z_mass_neg = z_mass <= wpfloat(0.0)

    in_bounds_m0 = k_half >= slevp1_ti + 1
    in_bounds_m1 = k_half >= slevp1_ti + 2
    in_bounds_m2 = k_half >= slevp1_ti + 3
    in_bounds_m3 = k_half >= slevp1_ti + 4

    p_cellmass_m0 = concat_where(
        dims.KHalfDim >= slevp1_ti + 1, p_cellmass_now(dims.KHalfDim - 0.5), wpfloat(0.0)
    )
    p_cellmass_m1 = concat_where(
        dims.KHalfDim >= slevp1_ti + 1, p_cellmass_now(dims.KHalfDim - 1.5), wpfloat(0.0)
    )
    p_cellmass_m2 = concat_where(
        dims.KHalfDim >= slevp1_ti + 2, p_cellmass_now(dims.KHalfDim - 2.5), wpfloat(0.0)
    )
    p_cellmass_m3 = concat_where(
        dims.KHalfDim >= slevp1_ti + 3, p_cellmass_now(dims.KHalfDim - 3.5), wpfloat(0.0)
    )
    p_cellmass_m4 = concat_where(
        dims.KHalfDim >= slevp1_ti + 4, p_cellmass_now(dims.KHalfDim - 4.5), wpfloat(0.0)
    )

    mass_gt_cellmass_m0 = where(z_mass_neg & in_bounds_m0, abs(z_mass) >= p_cellmass_m0, False)
    z_mass = z_mass + where(mass_gt_cellmass_m0, p_cellmass_m0, wpfloat(0.0))

    mass_gt_cellmass_m1 = mass_gt_cellmass_m0 & where(
        z_mass_neg & in_bounds_m1, abs(z_mass) >= p_cellmass_m1, False
    )
    z_mass = z_mass + where(mass_gt_cellmass_m1, p_cellmass_m1, wpfloat(0.0))

    mass_gt_cellmass_m2 = mass_gt_cellmass_m1 & where(
        z_mass_neg & in_bounds_m2, abs(z_mass) >= p_cellmass_m2, False
    )
    z_mass = z_mass + where(mass_gt_cellmass_m2, p_cellmass_m2, wpfloat(0.0))

    mass_gt_cellmass_m3 = mass_gt_cellmass_m2 & where(
        z_mass_neg & in_bounds_m3, abs(z_mass) >= p_cellmass_m3, False
    )
    z_mass = z_mass + where(mass_gt_cellmass_m3, p_cellmass_m3, wpfloat(0.0))

    p_cellmass_now_jks = concat_where(
        dims.KHalfDim > 0, p_cellmass_now(dims.KHalfDim - 0.5), wpfloat(0.0)
    )
    p_cellmass_now_jks = where(mass_gt_cellmass_m0, p_cellmass_m1, p_cellmass_now_jks)
    p_cellmass_now_jks = where(mass_gt_cellmass_m1, p_cellmass_m2, p_cellmass_now_jks)
    p_cellmass_now_jks = where(mass_gt_cellmass_m2, p_cellmass_m3, p_cellmass_now_jks)
    p_cellmass_now_jks = where(mass_gt_cellmass_m3, p_cellmass_m4, p_cellmass_now_jks)

    z_cfl = z_cfl - where(mass_gt_cellmass_m0, wpfloat(1.0), wpfloat(0.0))
    z_cfl = z_cfl - where(mass_gt_cellmass_m1, wpfloat(1.0), wpfloat(0.0))
    z_cfl = z_cfl - where(mass_gt_cellmass_m2, wpfloat(1.0), wpfloat(0.0))
    z_cfl = z_cfl - where(mass_gt_cellmass_m3, wpfloat(1.0), wpfloat(0.0))

    z_cflfrac = where(z_mass_neg, z_mass / p_cellmass_now_jks, wpfloat(0.0))
    z_cfl = z_cfl + where(abs(z_cflfrac) < wpfloat(1.0), z_cflfrac, wp_eps - wpfloat(1.0))

    return z_cfl


@gtx.field_operator
def _compute_ppm_courant_number(
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKHalfField[ta.wpfloat],
    k_half: fa.KHalfField[gtx.int32],
    slevp1_ti: gtx.int32,
    nlev: gtx.int32,
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
) -> fa.CellKHalfField[ta.wpfloat]:
    z_mass = p_dtime * p_mflx_contra_v

    cfl_below = _compute_courant_number_below(
        p_cellmass_now=p_cellmass_now,
        z_mass=z_mass,
        z_cfl=z_cfl,
        k_half=k_half,
        nlev=nlev,
        wp_eps=wp_eps,
    )
    cfl_above = _compute_courant_number_above(
        p_cellmass_now=p_cellmass_now,
        z_mass=z_mass,
        z_cfl=z_cfl,
        k_half=k_half,
        slevp1_ti=slevp1_ti,
        wp_eps=wp_eps,
    )

    z_cfl = cfl_below + cfl_above

    return z_cfl


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_courant_number(
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKHalfField[ta.wpfloat],
    k_half: fa.KHalfField[gtx.int32],
    slevp1_ti: gtx.int32,
    nlev: gtx.int32,
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_courant_number(
        p_mflx_contra_v=p_mflx_contra_v,
        p_cellmass_now=p_cellmass_now,
        z_cfl=z_cfl,
        k_half=k_half,
        slevp1_ti=slevp1_ti,
        nlev=nlev,
        wp_eps=wp_eps,
        p_dtime=p_dtime,
        out=z_cfl,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )


# TODO(nfarabullini, OngChia): this stencil has no test
# TODO(nfarabullini, OngChia): this stencil does not strictly match the fortran code


@gtx.field_operator
def _sum_neighbor_contributions(
    mask1: fa.CellKHalfField[bool],
    mask2: fa.CellKHalfField[bool],
    js: fa.CellKHalfField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    elev: gtx.int32,
) -> fa.CellKHalfField[ta.wpfloat]:
    js_eq0 = js == wpfloat("0.0")
    js_eq1 = js == wpfloat("1.0")
    js_eq2 = js == wpfloat("2.0")
    js_eq3 = js == wpfloat("3.0")
    js_eq4 = js == wpfloat("4.0")
    p_cc_k = p_cc(dims.KHalfDim + 0.5)

    p_cc_p0 = where(mask1 & js_eq0, p_cc_k, wpfloat("0.0"))
    p_cc_p1 = where(
        mask1 & js_eq1,
        concat_where(dims.KHalfDim < elev, p_cc(dims.KHalfDim + 1.5), p_cc_k),
        wpfloat("0.0"),
    )
    p_cc_p2 = where(
        mask1 & js_eq2,
        concat_where(dims.KHalfDim < elev - 1, p_cc(dims.KHalfDim + 2.5), p_cc_k),
        wpfloat("0.0"),
    )
    p_cc_p3 = where(
        mask1 & js_eq3,
        concat_where(dims.KHalfDim < elev - 2, p_cc(dims.KHalfDim + 3.5), p_cc_k),
        wpfloat("0.0"),
    )
    p_cc_p4 = where(
        mask1 & js_eq4,
        concat_where(dims.KHalfDim < elev - 3, p_cc(dims.KHalfDim + 4.5), p_cc_k),
        wpfloat("0.0"),
    )

    p_cc_m0 = where(
        mask2 & js_eq0,
        concat_where(dims.KHalfDim > 0, p_cc(dims.KHalfDim - 0.5), p_cc_k),
        wpfloat("0.0"),
    )
    p_cc_m1 = where(
        mask2 & js_eq1,
        concat_where(dims.KHalfDim > 1, p_cc(dims.KHalfDim - 1.5), p_cc_k),
        wpfloat("0.0"),
    )
    p_cc_m2 = where(
        mask2 & js_eq2,
        concat_where(dims.KHalfDim > 2, p_cc(dims.KHalfDim - 2.5), p_cc_k),
        wpfloat("0.0"),
    )
    p_cc_m3 = where(
        mask2 & js_eq3,
        concat_where(dims.KHalfDim > 3, p_cc(dims.KHalfDim - 3.5), p_cc_k),
        wpfloat("0.0"),
    )
    p_cc_m4 = where(
        mask2 & js_eq4,
        concat_where(dims.KHalfDim > 4, p_cc(dims.KHalfDim - 4.5), p_cc_k),
        wpfloat("0.0"),
    )

    p_cc_jks = (
        p_cc_p0
        + p_cc_p1
        + p_cc_p2
        + p_cc_p3
        + p_cc_p4
        + p_cc_m0
        + p_cc_m1
        + p_cc_m2
        + p_cc_m3
        + p_cc_m4
    )
    return p_cc_jks


@gtx.field_operator
def _compute_ppm_fractional_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKHalfField[ta.wpfloat],
    z_delta_q: fa.CellKField[ta.wpfloat],
    z_a1: fa.CellKField[ta.wpfloat],
    k_half: fa.KHalfField[gtx.int32],
    slev: gtx.int32,
    elev: gtx.int32,
    p_dtime: ta.wpfloat,
) -> fa.CellKHalfField[ta.wpfloat]:
    js = floor(abs(z_cfl))
    z_cflfrac = abs(z_cfl) - js
    z_cflfrac_nonzero = z_cflfrac != wpfloat("0.0")

    z_cfl_pos = z_cfl > wpfloat("0.0")
    z_cfl_neg = z_cfl < wpfloat("0.0")
    wsign = where(z_cfl_pos, wpfloat("1.0"), wpfloat("-1.0"))

    mask1 = z_cfl_pos & z_cflfrac_nonzero
    mask2 = z_cfl_neg & z_cflfrac_nonzero

    in_slev_bounds = astype(k_half, wpfloat) - js >= astype(slev, wpfloat)

    p_cc_jks = _sum_neighbor_contributions(mask1=mask1, mask2=mask2, js=js, p_cc=p_cc, elev=elev)
    p_cellmass_now_jks = _sum_neighbor_contributions(
        mask1=mask1, mask2=mask2, js=js, p_cc=p_cellmass_now, elev=elev
    )
    z_delta_q_jks = _sum_neighbor_contributions(
        mask1=mask1, mask2=mask2, js=js, p_cc=z_delta_q, elev=elev
    )
    z_a1_jks = _sum_neighbor_contributions(mask1=mask1, mask2=mask2, js=js, p_cc=z_a1, elev=elev)

    z_q_int = (
        p_cc_jks
        + wsign * (z_delta_q_jks * (wpfloat("1.0") - z_cflfrac))
        - z_a1_jks
        * (wpfloat("1.0") - wpfloat("3.0") * z_cflfrac + wpfloat("2.0") * z_cflfrac * z_cflfrac)
    )

    p_upflux = where(
        in_slev_bounds, wsign * p_cellmass_now_jks * z_cflfrac * z_q_int / p_dtime, wpfloat("0.0")
    )

    return p_upflux


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_fractional_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKHalfField[ta.wpfloat],
    z_delta_q: fa.CellKField[ta.wpfloat],
    z_a1: fa.CellKField[ta.wpfloat],
    p_upflux: fa.CellKHalfField[ta.wpfloat],
    k_half: fa.KHalfField[gtx.int32],
    slev: gtx.int32,
    elev: gtx.int32,
    p_dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_fractional_flux(
        p_cc=p_cc,
        p_cellmass_now=p_cellmass_now,
        z_cfl=z_cfl,
        z_delta_q=z_delta_q,
        z_a1=z_a1,
        k_half=k_half,
        slev=slev,
        elev=elev,
        p_dtime=p_dtime,
        out=p_upflux,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )


# TODO(nfarabullini, OngChia): this stencil has no test
# TODO(nfarabullini, OngChia): this stencil does not strictly match the fortran code


@gtx.field_operator
def _sum_neighbor_contributions_all(
    mask1: fa.CellKHalfField[bool],
    mask2: fa.CellKHalfField[bool],
    js: fa.CellKHalfField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    elev: gtx.int32,
) -> fa.CellKHalfField[ta.wpfloat]:
    js_gt0 = js >= wpfloat(0.0)
    js_gt1 = js >= wpfloat(1.0)
    js_gt2 = js >= wpfloat(2.0)
    js_gt3 = js >= wpfloat(3.0)
    js_gt4 = js >= wpfloat(4.0)

    prod_p0 = where(
        mask1 & js_gt0,
        concat_where(
            dims.KHalfDim < elev + 1,
            p_cc(dims.KHalfDim + 0.5) * p_cellmass_now(dims.KHalfDim + 0.5),
            wpfloat(0.0),
        ),
        wpfloat(0.0),
    )
    prod_p1 = where(
        mask1 & js_gt1,
        concat_where(
            dims.KHalfDim < elev,
            p_cc(dims.KHalfDim + 1.5) * p_cellmass_now(dims.KHalfDim + 1.5),
            wpfloat(0.0),
        ),
        wpfloat(0.0),
    )
    prod_p2 = where(
        mask1 & js_gt2,
        concat_where(
            dims.KHalfDim < elev - 1,
            p_cc(dims.KHalfDim + 2.5) * p_cellmass_now(dims.KHalfDim + 2.5),
            wpfloat(0.0),
        ),
        wpfloat(0.0),
    )
    prod_p3 = where(
        mask1 & js_gt3,
        concat_where(
            dims.KHalfDim < elev - 2,
            p_cc(dims.KHalfDim + 3.5) * p_cellmass_now(dims.KHalfDim + 3.5),
            wpfloat(0.0),
        ),
        wpfloat(0.0),
    )
    prod_p4 = where(
        mask1 & js_gt4,
        concat_where(
            dims.KHalfDim < elev - 3,
            p_cc(dims.KHalfDim + 4.5) * p_cellmass_now(dims.KHalfDim + 4.5),
            wpfloat(0.0),
        ),
        wpfloat(0.0),
    )
    prod_m0 = where(
        mask2 & js_gt0,
        concat_where(
            dims.KHalfDim > 0,
            p_cc(dims.KHalfDim - 0.5) * p_cellmass_now(dims.KHalfDim - 0.5),
            wpfloat(0.0),
        ),
        wpfloat(0.0),
    )
    prod_m1 = where(
        mask2 & js_gt1,
        concat_where(
            dims.KHalfDim > 1,
            p_cc(dims.KHalfDim - 1.5) * p_cellmass_now(dims.KHalfDim - 1.5),
            wpfloat(0.0),
        ),
        wpfloat(0.0),
    )
    prod_m2 = where(
        mask2 & js_gt2,
        concat_where(
            dims.KHalfDim > 2,
            p_cc(dims.KHalfDim - 2.5) * p_cellmass_now(dims.KHalfDim - 2.5),
            wpfloat(0.0),
        ),
        wpfloat(0.0),
    )
    prod_m3 = where(
        mask2 & js_gt3,
        concat_where(
            dims.KHalfDim > 3,
            p_cc(dims.KHalfDim - 3.5) * p_cellmass_now(dims.KHalfDim - 3.5),
            wpfloat(0.0),
        ),
        wpfloat(0.0),
    )
    prod_m4 = where(
        mask2 & js_gt4,
        concat_where(
            dims.KHalfDim > 4,
            p_cc(dims.KHalfDim - 4.5) * p_cellmass_now(dims.KHalfDim - 4.5),
            wpfloat(0.0),
        ),
        wpfloat(0.0),
    )

    prod_jks = (
        prod_p0
        + prod_p1
        + prod_p2
        + prod_p3
        + prod_p4
        + prod_m0
        + prod_m1
        + prod_m2
        + prod_m3
        + prod_m4
    )
    return prod_jks


@gtx.field_operator
def _compute_ppm_integer_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKHalfField[ta.wpfloat],
    p_upflux: fa.CellKHalfField[ta.wpfloat],
    k_half: fa.KHalfField[gtx.int32],
    slev: gtx.int32,
    elev: gtx.int32,
    p_dtime: ta.wpfloat,
) -> fa.CellKHalfField[ta.wpfloat]:
    js = floor(abs(z_cfl)) - wpfloat(1.0)

    z_cfl_pos = z_cfl > wpfloat(0.0)
    z_cfl_neg = z_cfl < wpfloat(0.0)
    wsign = where(z_cfl_pos, wpfloat(1.0), wpfloat(-1.0))

    in_slev_bounds = astype(k_half, wpfloat) - js >= astype(slev, wpfloat)

    p_cc_cellmass_now_jks = _sum_neighbor_contributions_all(
        mask1=z_cfl_pos,
        mask2=z_cfl_neg,
        js=js,
        p_cc=p_cc,
        p_cellmass_now=p_cellmass_now,
        elev=elev,
    )

    z_iflx = wsign * p_cc_cellmass_now_jks

    p_upflux = p_upflux + where(in_slev_bounds, z_iflx / p_dtime, wpfloat(0.0))

    return p_upflux


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_integer_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKHalfField[ta.wpfloat],
    p_upflux: fa.CellKHalfField[ta.wpfloat],
    k_half: fa.KHalfField[gtx.int32],
    slev: gtx.int32,
    elev: gtx.int32,
    p_dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_integer_flux(
        p_cc=p_cc,
        p_cellmass_now=p_cellmass_now,
        z_cfl=z_cfl,
        p_upflux=p_upflux,
        k_half=k_half,
        slev=slev,
        elev=elev,
        p_dtime=p_dtime,
        out=p_upflux,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_ppm_parabola_coefficients(
    z_face_up: fa.CellKField[ta.wpfloat],
    z_face_low: fa.CellKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    z_delta_q = wpfloat(0.5) * (z_face_up - z_face_low)
    z_a1 = p_cc - wpfloat(0.5) * (z_face_up + z_face_low)

    return z_delta_q, z_a1


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_parabola_coefficients(
    z_face_up: fa.CellKField[ta.wpfloat],
    z_face_low: fa.CellKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    z_delta_q: fa.CellKField[ta.wpfloat],
    z_a1: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_parabola_coefficients(
        z_face_up=z_face_up,
        z_face_low=z_face_low,
        p_cc=p_cc,
        out=(z_delta_q, z_a1),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_ppm_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    k: fa.KHalfField[gtx.int32],
    slev: gtx.int32,
    slevp1_ti: gtx.int32,
    elev: gtx.int32,
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    itype_vlimit: gtx.int32,
) -> fa.CellKHalfField[ta.wpfloat]:
    z_cfl = broadcast(wpfloat(0.0), (dims.CellDim, dims.KHalfDim))
    z_cfl = concat_where(
        (dims.KHalfDim > 0) & (dims.KHalfDim < elev + 1),
        _compute_ppm_courant_number(
            p_mflx_contra_v=p_mflx_contra_v,
            p_cellmass_now=p_cellmass_now,
            z_cfl=z_cfl,
            k_half=k,
            slevp1_ti=slevp1_ti,
            nlev=elev,
            wp_eps=wp_eps,
            p_dtime=p_dtime,
        ),
        z_cfl,
    )
    z_slope_raw = concat_where(
        (dims.KDim > 0) & (dims.KDim <= elev),
        _compute_ppm_slope(p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now, elev=elev),
        broadcast(wpfloat(0.0), (dims.CellDim, dims.KDim)),
    )
    z_slope = (
        _limit_vertical_slope_semi_monotonically(p_cc=p_cc, z_slope=z_slope_raw, elev=elev)
        if (itype_vlimit == VerticalAdvectionLimiter.SEMI_MONOTONIC)
        else z_slope_raw
    )
    p_face = concat_where(
        dims.KHalfDim == 0,
        p_cc(dims.KHalfDim + 0.5),
        concat_where(
            dims.KHalfDim < elev + 1,
            concat_where(
                (dims.KHalfDim > 1) & (dims.KHalfDim < elev),
                _compute_ppm_quartic_face_values(
                    p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now, z_slope=z_slope
                ),
                _compute_ppm_quadratic_face_values(p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now),
            ),
            p_cc(dims.KHalfDim - 0.5),
        ),
    )
    l_limit = (
        _compute_vertical_parabola_limiter_condition(p_face=p_face, p_cc=p_cc)
        if itype_vlimit == VerticalAdvectionLimiter.SEMI_MONOTONIC
        else broadcast(0, (dims.CellDim, dims.KDim))
    )
    z_face_up, z_face_low = _limit_vertical_parabola_semi_monotonically(
        l_limit=l_limit, p_face=p_face, p_cc=p_cc
    )
    z_delta_q, z_a1 = _compute_ppm_parabola_coefficients(
        z_face_up=z_face_up, z_face_low=z_face_low, p_cc=p_cc
    )
    p_upflux = _compute_ppm_fractional_flux(
        p_cc=p_cc,
        p_cellmass_now=p_cellmass_now,
        z_cfl=z_cfl,
        z_delta_q=z_delta_q,
        z_a1=z_a1,
        k_half=k,
        slev=slev,
        elev=elev,
        p_dtime=p_dtime,
    )
    p_upflux = _compute_ppm_integer_flux(
        p_cc=p_cc,
        p_cellmass_now=p_cellmass_now,
        z_cfl=z_cfl,
        p_upflux=p_upflux,
        k_half=k,
        slev=slev,
        elev=elev,
        p_dtime=p_dtime,
    )
    return concat_where(
        (dims.KHalfDim > 0) & (dims.KHalfDim < elev + 1),
        p_upflux,
        broadcast(wpfloat(0.0), (dims.CellDim, dims.KHalfDim)),
    )
