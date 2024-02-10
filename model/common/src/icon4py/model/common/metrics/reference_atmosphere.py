# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next import Field, GridType, field_operator, program
from gt4py.next.ffront.fbuiltins import exp, int32, log, neighbor_sum

from icon4py.model.common.dimension import CellDim, KDim, E2CDim, EdgeDim, E2C
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _cell_2_edge_interpolation(
    in_field: Field[[CellDim, KDim], wpfloat], coeff: Field[[EdgeDim, E2CDim], wpfloat]
) -> Field[[EdgeDim, KDim], wpfloat]:
    """
    Interpolate a Cell Field to Edges.

    There is a special handling of lateral boundary edges in `subroutine cells2edges_scalar`
    where the value is set to the one valid in_field value without multiplication by coeff.
    That essentially means: the skip value neighbor in the neighbor_sum is skipped and coeff needs to
    be 1 for this Edge index.
    """
    return neighbor_sum(in_field(E2C) * coeff, axis=E2CDim)


@program(grid_type=GridType.UNSTRUCTURED)
def cell_2_edge_interpolation(
    in_field: Field[[CellDim, KDim], wpfloat],
    coeff: Field[[EdgeDim, E2CDim], wpfloat],
    out_field: Field[[EdgeDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _cell_2_edge_interpolation(
        in_field,
        coeff,
        out=out_field,
        domain={EdgeDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def _compute_reference_atmosphere_edge_fields(
    z_me: Field[[EdgeDim, KDim], wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
) -> tuple[Field[[EdgeDim, KDim], wpfloat], Field[[EdgeDim, KDim], wpfloat]]:
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
    z_me: Field[[EdgeDim, KDim], wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    rho_ref_me: Field[[EdgeDim, KDim], wpfloat],
    theta_ref_me: Field[[EdgeDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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
        domain={EdgeDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def compute_z_temp(
    z_mc: Field[[CellDim, KDim], wpfloat],
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
) -> Field[[CellDim, KDim], wpfloat]:
    denom = t0sl_bg - del_t_bg
    z_temp = denom + del_t_bg * exp(-z_mc / h_scal_bg)
    return z_temp


@field_operator
def compute_z_aux1_cell(
    z_mc: Field[[CellDim, KDim], wpfloat],
    p0sl_bg: wpfloat,
    grav: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
) -> Field[[CellDim, KDim], wpfloat]:
    denom = t0sl_bg - del_t_bg
    logval = log((exp(z_mc / h_scal_bg) * denom + del_t_bg) / t0sl_bg)
    return p0sl_bg * exp(-grav / rd * h_scal_bg / denom * logval)


@field_operator
def _compute_reference_atmosphere_cell_fields(
    z_mc: Field[[CellDim, KDim], wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
) -> tuple[
    Field[[CellDim, KDim], wpfloat],
    Field[[CellDim, KDim], wpfloat],
    Field[[CellDim, KDim], wpfloat],
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
    z_height: Field[[CellDim, KDim], wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    exner_ref_mc: Field[[CellDim, KDim], wpfloat],
    rho_ref_mc: Field[[CellDim, KDim], wpfloat],
    theta_ref_mc: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """
        Calculate reference atmosphere fields on full levels.

    Args:
        z_height: geometric height
        p0ref: reference pressure for exner function [Pa]
        p0sl_bg: sea level pressuer [Pa]
        grav: gravitational constant [m/s^2]
        cpd: specific heat at constant pressure [J/K/kg]
        rd: gas constant for dry air [J/K/kg]
        h_scal_bg: height scale [m]
        t0sl_bg: sea level temperature [K]
        del_t_bg: temperature difference between sea level and asymptotic stratospheric temperature
        exner_ref_mc: (output) reference exner pressure on full level mass points
        rho_ref_mc: (output) reference density on full level mass points
        theta_ref_mc: (output) reference potential temperature on full level mass points
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
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@field_operator
def compute_d_exner_dz_ref_ic(
    theta_ref_ic: Field[[CellDim, KDim], wpfloat], grav: wpfloat, cpd: wpfloat
) -> Field[[CellDim, KDim], wpfloat]:
    """
    Calculate first vertical derivative of reference Exner pressure, half level mass points
    Args:
        theta_ref_ic: reference potential temperature
        grav: gravitational constant [m/s^2]
        cpd: specific heat at constant pressure [J/K/kg]

    Returns: first vertical derivative of reference exner pressure
    """
    return -grav / (cpd * theta_ref_ic)
