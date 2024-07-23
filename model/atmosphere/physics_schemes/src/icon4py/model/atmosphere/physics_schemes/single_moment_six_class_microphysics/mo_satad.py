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

"""
Changes.

- Only implemented Tetens (ipsat = 1). Dropped Murphy-Koop.
- Harmonized name of constants
- Only implementend gpu version. Maybe further optimizations possible for CPU (check original code)

TODO:
1. Implement Newtonian iteration! -> Needs fixted-size for loop feature in GT4Py

Comment from FORTRAN version:
- Suggested by U. Blahak: Replace pres_sat_water, pres_sat_ice and spec_humi by
lookup tables in mo_convect_tables. Bit incompatible change!
"""
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, abs, exp, maximum, where

from icon4py.atm_phy_schemes.mo_convect_tables import conv_table
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.shared.mo_physical_constants import phy_const
from icon4py.model.common.settings import xp
from gt4py.next.program_processors.runners.gtfn_cpu import (
    run_gtfn,
    run_gtfn_cached,
    run_gtfn_imperative,
)

@field_operator
def _latent_heat_vaporization(
    t: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:
    """Return latent heat of vaporization.

    Computed as internal energy and taking into account Kirchoff's relations
    """
    # specific heat of water vapor at constant pressure (Landolt-Bornstein)
    #cp_v = 1850.0

    return (
        phy_const.alv
        + (1850.0 - phy_const.clw) * (t - phy_const.tmelt)
        - phy_const.rv * t
    )


@field_operator
def _sat_pres_water(t: Field[[CellDim, KDim], float]) -> Field[[CellDim, KDim], float]:
    """Return saturation water vapour pressure."""
    return conv_table.c1es * exp(
        conv_table.c3les * (t - phy_const.tmelt) / (t - conv_table.c4les)
    )


@field_operator
def _qsat_rho(
    t: Field[[CellDim, KDim], float], rho: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:
    """Return specific humidity at water saturation (with respect to flat surface)."""
    return _sat_pres_water(t) / (rho * phy_const.rv * t)


@field_operator
def _dqsatdT_rho(
    t: Field[[CellDim, KDim], float], zqsat: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:
    """
    Return partial derivative of the specific humidity at water saturation.

    Computed with respect to the temperature at constant total density.
    """
    beta = conv_table.c5les / (t - conv_table.c4les) ** 2 - 1.0 / t
    return beta * zqsat


@field_operator
def _second_onwards_newtonian_iteration(
    t: Field[[CellDim, KDim], float],
    t2: Field[[CellDim, KDim], float],
    qv: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
    lwdocvd: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:
    tol = 1e-3  # tolerance for iteration

    ft = t2 - t + lwdocvd * (_qsat_rho(t2, rho) - qv)
    dft = 1.0 + lwdocvd * _dqsatdT_rho(t2, _qsat_rho(t2, rho))

    return where(abs(t2 - t) > tol, t2 - ft / dft, t2)


@field_operator
def _first_newtonian_iteration(
    t: Field[[CellDim, KDim], float],
    qv: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
    lwdocvd: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:

    ft = lwdocvd * (_qsat_rho(t, rho) - qv)
    dft = 1.0 + lwdocvd * _dqsatdT_rho(t, _qsat_rho(t, rho))

    return t - ft / dft


@field_operator
def _compute_temperature_in_first_satad_iteration(
    t: Field[[CellDim, KDim], float],
    qv: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
    temperature_after_all_qc_evaporated: Field[[CellDim, KDim], float],
    subsaturated_mask: Field[[CellDim, KDim], bool],
    lwdocvd: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:
    t = where(
        subsaturated_mask,
        # If all cloud water evaporates, no newtonian iteration ncessary
        temperature_after_all_qc_evaporated,
        _first_newtonian_iteration(t, qv, rho, lwdocvd),
    )
    return t

@field_operator
def _compute_temperature_from_second_satad_iteration(
    t: Field[[CellDim, KDim], float],
    t2: Field[[CellDim, KDim], float],
    qv: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
    temperature_after_all_qc_evaporated: Field[[CellDim, KDim], float],
    subsaturated_mask: Field[[CellDim, KDim], bool],
    lwdocvd: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:
    t = where(
        subsaturated_mask,
        # If all cloud water evaporates, no newtonian iteration ncessary
        temperature_after_all_qc_evaporated,
        _second_onwards_newtonian_iteration(t, t2, qv, rho, lwdocvd),
    )
    return t

@field_operator
def _update_qv_qc_in_satad(
    t: Field[[CellDim, KDim], float],
    qv: Field[[CellDim, KDim], float],
    qc: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
    subsaturated_mask: Field[[CellDim, KDim], bool]
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    # Local treshold
    zqwmin = 1e-20
    qv, qc = where(
        subsaturated_mask,
        (qv + qc, 0.0),
        (_qsat_rho(t, rho), maximum(qv + qc - _qsat_rho(t, rho), zqwmin)),
    )

    return qv, qc
@field_operator
def _compute_subsaturated_mask(
    t: Field[[CellDim, KDim], float],
    qv: Field[[CellDim, KDim], float],
    qc: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], bool], Field[[CellDim, KDim], float]]:
    """
    Adjust saturation at each grid point.

    Synopsis:
    Saturation adjustment condenses/evaporates specific humidity (qv) into/from
    cloud water content (qc) such that a gridpoint is just saturated. Temperature (t)
    is adapted accordingly and pressure adapts itself in ICON.

    Method:
    Saturation adjustment at constant total density (adjustment of T and p accordingly)
    assuming chemical equilibrium of water and vapor. For the heat capacity of
    of the total system (dry air, vapor, and hydrometeors) the value of dry air
    is taken, which is a common approximation and introduces only a small error.

    Originally inspirered from satad_v_3D_gpu of ICON release 2.6.4.
    """

    temperature_after_all_qc_evaporated = (
        t - _latent_heat_vaporization(t) / phy_const.cvd * qc
    )

    # Check, which points will still be subsaturated even after evaporating all cloud water.
    subsaturated_mask = qv + qc <= _qsat_rho(temperature_after_all_qc_evaporated, rho)

    # Remains const. during iteration
    lwdocvd = _latent_heat_vaporization(t) / phy_const.cvd

    return temperature_after_all_qc_evaporated, subsaturated_mask, lwdocvd

def saturation_adjustment(
    grid: icon_grid.IconGrid,
    t: Field[[CellDim, KDim], float],
    qv: Field[[CellDim, KDim], float],
    qc: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
)-> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    """
    Adjust saturation at each grid point.

    Synopsis:
    Saturation adjustment condenses/evaporates specific humidity (qv) into/from
    cloud water content (qc) such that a gridpoint is just saturated. Temperature (t)
    is adapted accordingly and pressure adapts itself in ICON.

    Method:
    Saturation adjustment at constant total density (adjustment of T and p accordingly)
    assuming chemical equilibrium of water and vapor. For the heat capacity of
    of the total system (dry air, vapor, and hydrometeors) the value of dry air
    is taken, which is a common approximation and introduces only a small error.

    Originally inspirered from satad_v_3D_gpu of ICON release 2.6.4.
    """

    temperature_after_all_qc_evaporated, subsaturated_mask, lwdocvd = _compute_subsaturated_mask(t, qv, qc, rho)

    t2 = _compute_temperature_in_first_satad_iteration(t, qv, qc, rho, temperature_after_all_qc_evaporated, subsaturated_mask, lwdocvd)

    for _ in range(10):
        if (xp.abs(t2.ndarray - t.ndarray) > 1.e-3).any():
            t2 = _compute_temperature_from_second_satad_iteration(
                t,
                t2,
                qv,
                qc,
                rho,
                temperature_after_all_qc_evaporated,
                subsaturated_mask,
                lwdocvd
            )
        else:
            break
    new_qv, new_qc = _update_qv_qc_in_satad(t, qv, qc, rho, subsaturated_mask)


#backend=run_gtfn
@program()
def satad(
    qv: Field[[CellDim, KDim], float],
    qc: Field[[CellDim, KDim], float],
    t: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
):

    _satad(qv, qc, t, rho, out=(t, qv, qc))
