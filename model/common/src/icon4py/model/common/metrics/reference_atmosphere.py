# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import GridType, field_operator, program
from gt4py.next.ffront.fbuiltins import astype, exp, log

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_reference_atmosphere_edge_fields(
    z_me: fa.EdgeKField[wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    denom = t0sl_bg - del_t_bg
    exp_z_me = exp(z_me / h_scal_bg)
    logval = log((exp_z_me * denom + del_t_bg) / t0sl_bg)
    z_aux_1 = p0sl_bg * exp(-grav / rd * h_scal_bg / denom * logval)
    z_temp = denom + del_t_bg * exp(-z_me / h_scal_bg)
    rho_ref_me = z_aux_1 / (rd * z_temp)
    rd_o_cpd = rd / cpd
    theta_ref_me = z_temp / (z_aux_1 / p0ref) ** rd_o_cpd
    return (rho_ref_me, theta_ref_me)


@program(grid_type=GridType.UNSTRUCTURED)
def compute_reference_atmosphere_edge_fields(
    z_me: fa.EdgeKField[wpfloat],
    rho_ref_me: fa.EdgeKField[wpfloat],
    theta_ref_me: fa.EdgeKField[wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_reference_atmosphere_edge_fields(
        z_me,
        p0ref,
        p0sl_bg,
        grav,
        cpd,
        rd,
        h_scal_bg,
        t0sl_bg,
        del_t_bg,
        out=(rho_ref_me, theta_ref_me),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def compute_z_temp(
    z_mc: fa.CellKField[wpfloat],
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
) -> fa.CellKField[wpfloat]:
    denom = t0sl_bg - del_t_bg
    z_temp = denom + del_t_bg * exp(-z_mc / h_scal_bg)
    return z_temp


@field_operator
def compute_z_aux1_cell(
    z_mc: fa.CellKField[wpfloat],
    p0sl_bg: wpfloat,
    grav: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
) -> fa.CellKField[wpfloat]:
    denom = t0sl_bg - del_t_bg
    logval = log((exp(z_mc / h_scal_bg) * denom + del_t_bg) / t0sl_bg)
    return p0sl_bg * exp(-grav / rd * h_scal_bg / denom * logval)


@field_operator
def _compute_reference_atmosphere_cell_fields(
    z_mc: fa.CellKField[wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    z_aux1 = compute_z_aux1_cell(
        z_mc=z_mc,
        p0sl_bg=p0sl_bg,
        grav=grav,
        rd=rd,
        h_scal_bg=h_scal_bg,
        t0sl_bg=t0sl_bg,
        del_t_bg=del_t_bg,
    )

    rd_o_cpd = rd / cpd
    exner_ref_mc = (z_aux1 / p0ref) ** rd_o_cpd
    z_temp = compute_z_temp(z_mc=z_mc, del_t_bg=del_t_bg, t0sl_bg=t0sl_bg, h_scal_bg=h_scal_bg)
    rho_ref_mc = z_aux1 / (rd * z_temp)
    theta_ref_mc = z_temp / exner_ref_mc
    return (
        theta_ref_mc,
        exner_ref_mc,
        rho_ref_mc,
    )


@program(grid_type=GridType.UNSTRUCTURED)
def compute_reference_atmosphere_cell_fields(
    z_height: fa.CellKField[wpfloat],
    exner_ref_mc: fa.CellKField[wpfloat],
    rho_ref_mc: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
        Calculate reference atmosphere fields on full levels.

    Args:
        z_height: geometric height
        exner_ref_mc: (output) reference exner pressure on full level mass points
        rho_ref_mc: (output) reference density on full level mass points
        theta_ref_mc: (output) reference potential temperature on full level mass points
        p0ref: reference pressure for exner function [Pa]
        p0sl_bg: sea level pressuer [Pa]
        grav: gravitational constant [m/s^2]
        cpd: specific heat at constant pressure [J/K/kg]
        rd: gas constant for dry air [J/K/kg]
        h_scal_bg: height scale [m]
        t0sl_bg: sea level temperature [K]
        del_t_bg: temperature difference between sea level and asymptotic stratospheric temperature
        horizontal_start:int32 start index of horizontal domain
        horizontal_end:int32 end index of horizontal domain
        vertical_start:int32 start index of vertical domain
        vertical_end:int32 end index of vertical domain
    """
    _compute_reference_atmosphere_cell_fields(
        z_height,
        p0ref,
        p0sl_bg,
        grav,
        cpd,
        rd,
        h_scal_bg,
        t0sl_bg,
        del_t_bg,
        out=(theta_ref_mc, exner_ref_mc, rho_ref_mc),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def compute_d_exner_dz_ref_ic(
    theta_ref_ic: fa.CellKField[wpfloat], grav: wpfloat, cpd: wpfloat
) -> fa.CellKField[wpfloat]:
    """
    Calculate first vertical derivative of reference Exner pressure, half level mass points.

    Args:
        theta_ref_ic: reference potential temperature
        grav: gravitational constant [m/s^2]
        cpd: specific heat at constant pressure [J/K/kg]

    Returns: first vertical derivative of reference exner pressure
    """
    return -grav / (cpd * theta_ref_ic)


@field_operator
def _compute_d2dexdz2_fac_mc(
    theta_ref_mc: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    exner_ref_mc: fa.CellKField[vpfloat],
    z_mc: fa.CellKField[wpfloat],
    cpd: wpfloat,
    grav: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    """
    Compute vertical derivative of d_exner_dz/theta_ref for full level mass points.

    (d2dexdz2_fac1_mc and d2dexdz2_fac2_mc factors in mo_vertical_grid.f90)

    Args:
        theta_ref_mc: reference Potential temperature, full level mass points
        inv_ddqz_z_full: inverse layer thickness (for runtime optimization)
        exner_ref_mc: reference Exner pressure, full level mass points
        z_mc: geometric height defined on full levels
        cpd: Specific heat at constant pressure [J/K/kg]
        grav: average gravitational acceleration
        del_t_bg: difference between sea level temperature and asymptotic stratospheric temperature
        h_scal_bg: height scale for reference atmosphere [m]

    Returns:
        fac1: first vertical derivative of reference Exner pressure, full level mass points, divided by theta_ref
        fac2: vertical derivative of d_exner_dz/theta_ref, full level mass points


    """
    del_t_bg = astype(del_t_bg, vpfloat)
    cpd = astype(cpd, vpfloat)
    grav = astype(grav, vpfloat)
    h_scal_bg = astype(h_scal_bg, vpfloat)
    z_mc = astype(z_mc, vpfloat)
    fac1 = -grav / (cpd * theta_ref_mc**2) * inv_ddqz_z_full
    fac2 = (
        2.0
        * grav
        / (cpd * theta_ref_mc**3)
        * (grav / cpd - del_t_bg / h_scal_bg * exp(-z_mc / h_scal_bg))
        / exner_ref_mc
    )

    return fac1, fac2


@program(grid_type=GridType.UNSTRUCTURED)
def compute_d2dexdz2_fac_mc(
    theta_ref_mc: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    exner_ref_mc: fa.CellKField[vpfloat],
    z_mc: fa.CellKField[wpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[vpfloat],
    cpd: float,
    grav: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_d2dexdz2_fac_mc(
        theta_ref_mc=theta_ref_mc,
        inv_ddqz_z_full=inv_ddqz_z_full,
        exner_ref_mc=exner_ref_mc,
        z_mc=z_mc,
        cpd=cpd,
        grav=grav,
        del_t_bg=del_t_bg,
        h_scal_bg=h_scal_bg,
        out=(d2dexdz2_fac1_mc, d2dexdz2_fac2_mc),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
