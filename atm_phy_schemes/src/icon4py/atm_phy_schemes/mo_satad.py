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
- Is where statement the nicest possible syntax for the IF-ELSE below? Also, where currently can only return 1 Field(
(a) Do np.where(fun1, func2) implementation
(b) How would algorithm look in local view? In current localview frontend.
(c) ...
- Implement Newtonian iteration! -> Needs fixted-size for loop feature in GT4Py
(0) For loop outside (maybe low priority)
(a) Unroll by hand
(b) Naive unroll of compile time FOR, maybe optimize
(c) Tracing
- Should pre-computations be removed? E.g, lwdocvd and qwa ???
(a) Currently only partly optimized. Should be fine if in same @program
- Assignment operators: --> WONTFIX for now, since we want to behave like numpy.
- Support for docstings inside field_operators. For now: Place all docstrings outside of field_operators as they are not supported inside.
--> Trivial to implement
- What to do with


Nice to have:
- Document constants such that description appears when hovering over symbol in IDE's -> mo_physical_constants.py et al
-> DL:adapt code

Comment from FORTRAN version:
- Suggested by U. Blahak: Replace pres_sat_water, pres_sat_ice and spec_humi by
lookup tables in mo_convect_tables. Bit incompatible change!


# Results: Big ticket features
0. Docstings in field_operator (Nikki)
1. start planning for loop or recursion (Linus)
2. start thinking about multiple returns form where (Nikki)
3. local, global, compile-time constants (Peter)


"""
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, exp, where, abs, maximum

# from icon4py.atm_phy_schemes.mo_convect_tables import c1es, c3les, c4les, c5les
from icon4py.common.dimension import CellDim, KDim

# from icon4py.shared.mo_physical_constants import alv, clw, cvd, rv, tmelt

# # TODO: Local constants What to do with these?
# cp_v = 1850.0  # specific heat of water vapor at constant pressure (Landolt-Bornstein)
# ci = 2108.0  # specific heat of ice

# tol = 1e-3
# maxiter = 10  #
# zqwmin = 1e-20

# TODO: Docstrings will crash field_Operators
"""
Return latent heat of vaporization.

Computed as internal energy and taking into account Kirchoff's relations
"""


@field_operator
def _latent_heat_vaporization(
    t: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:

    # TODO: Remove later
    alv = 2.5008e6
    tmelt = 273.15
    rv = 461.51
    cpd = 1004.64
    rcpl = 3.1733
    clw = (rcpl + 1.0) * cpd
    cp_v = 1850.0

    return alv + (cp_v - clw) * (t - tmelt) - rv * t


"""Return saturation water vapour pressure."""


@field_operator
def _sat_pres_water(t: Field[[CellDim, KDim], float]) -> Field[[CellDim, KDim], float]:
    # TODO: Remove later
    tmelt = 273.15
    c1es = 610.78
    c3les = 17.269
    c4les = 35.86

    return c1es * exp(c3les * (t - tmelt) / (t - c4les))


"""Return specific humidity at water saturation (with respect to flat surface)."""


@field_operator
def _qsat_rho(
    t: Field[[CellDim, KDim], float], rho: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:

    rv = 461.51  # TODO: Remove

    return _sat_pres_water(t) / (rho * rv * t)


"""
Return partial derivative of the specific humidity at water saturation.

Computed with respect to the temperature at constant total density.
"""


@field_operator
def _dqsatdT_rho(
    t: Field[[CellDim, KDim], float], zqsat: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:

    # TODO: Remove later
    tmelt = 273.15
    c3les = 17.269
    c4les = 35.86
    c5les = c3les * (tmelt - c4les)

    beta = c5les / (t - c4les) ** 2 - 1.0 / t
    return beta * zqsat


@field_operator
def _newtonian_for_body(
    t: Field[[CellDim, KDim], float],
    qv: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
    lwdocvd: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    # TODO: RENAME and test if identical. Double check with Hannes
    # Save the starting value of temperature t
    twork = t

    fT = twork - t + lwdocvd * (_qsat_rho(twork, rho) - qv)
    dfT = 1.0 + lwdocvd * _dqsatdT_rho(twork, _qsat_rho(twork, rho))

    return twork - fT / dfT


@field_operator
def _conditional_newtonian_for_body(
    t: Field[[CellDim, KDim], float],
    qv: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
    lwdocvd: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    # TODO: RENAME and test if identical. Double check with Hannes
    # Save the starting value of temperature t
    twork = t

    # Here we still have to iterate ...
    tworkold = twork

    fT = twork - t + lwdocvd * (_qsat_rho(twork, rho) - qv)
    dfT = 1.0 + lwdocvd * _dqsatdT_rho(twork, _qsat_rho(twork, rho))

    tol = 1e-3  # TODO: Remove
    return where(abs(twork - tworkold) > tol, twork - fT / dfT, tworkold)


@field_operator
def _newtonian_iteration_t(
    t: Field[[CellDim, KDim], float],
    qv: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:

    # TODO: Remove
    rd = 287.04
    cpd = 1004.64
    cvd = cpd - rd

    lwdocvd = _latent_heat_vaporization(t) / cvd

    # for _ in range(1, maxiter):
    twork = _newtonian_for_body(t, qv, rho, lwdocvd)
    twork = _conditional_newtonian_for_body(twork, qv, rho, lwdocvd)
    twork = _conditional_newtonian_for_body(twork, qv, rho, lwdocvd)
    twork = _conditional_newtonian_for_body(twork, qv, rho, lwdocvd)
    twork = _conditional_newtonian_for_body(twork, qv, rho, lwdocvd)
    twork = _conditional_newtonian_for_body(twork, qv, rho, lwdocvd)
    twork = _conditional_newtonian_for_body(twork, qv, rho, lwdocvd)
    twork = _conditional_newtonian_for_body(twork, qv, rho, lwdocvd)
    twork = _conditional_newtonian_for_body(twork, qv, rho, lwdocvd)
    return _conditional_newtonian_for_body(twork, qv, rho, lwdocvd)


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


@field_operator
def _satad(
    qv: Field[[CellDim, KDim], float],
    qc: Field[[CellDim, KDim], float],
    t: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:

    # TODO: Remove
    rd = 287.04
    cpd = 1004.64
    cvd = cpd - rd
    zqwmin = 1e-20

    tTest = t - _latent_heat_vaporization(t) / cvd * qc

    # check, which points will still be subsaturated even after evaporating
    # all cloud water.For these gridpoints Newton iteration is not necessary.
    subsaturated = qv + qc <= _qsat_rho(t, rho)

    qv = where(subsaturated, qv + qc, _qsat_rho(t, rho))
    qc = where(subsaturated, 0.0, maximum(qv + qc - _qsat_rho(t, rho), zqwmin))
    t = where(subsaturated, tTest, _newtonian_iteration_t(t, qv, rho))

    return t, qv, qc


# DL: Do we actually need the program?
@program
def satad(
    qv: Field[[CellDim, KDim], float],
    qc: Field[[CellDim, KDim], float],
    t: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
):
    _satad(qv, qc, t, rho, out=(t, qv, qc))
