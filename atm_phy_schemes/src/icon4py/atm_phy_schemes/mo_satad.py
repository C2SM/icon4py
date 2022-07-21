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
- Need math-builtins in GT4Py (e.g., exp) -> Workinprogress
- Need support for retrun of multiple fields in field_operators and programs, e.g., return tuple -> Workinprogress
- Can we call field_operator from a field_operator? -> Workinprogress
- Is where statement the nicest possible syntax for the IF-ELSE below? Also, where currently can only return 1 Field
- Implement Newtonian iteration! -> Needs fixted-size for loop feature in GT4Py
- Should pre-computations be removed? E.g, lwdocvd and qwa ???
- Assignment operators: Suppress/blacklist with (linters?) until supported?
- Support for docstings inside field_operators. For now: Place all docstrings outside of field_operators as they are not supported inside.
- What to do with local constants?

Nice to have:
- Document constants such that description appears when hovering over symbol in IDE's -> mo_physical_constants.py et al

Comment from FORTRAN version:
- Suggested by U. Blahak: Replace pres_sat_water, pres_sat_ice and spec_humi by
lookup tables in mo_convect_tables. Bit incompatible change!
"""
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field

from icon4py.atm_phy_schemes.mo_convect_tables import (
    c1es,
    c3les,
    c4les,
    c5les,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.shared.mo_physical_constants import (
    alv,
    clw,
    cvd,
    rv,
    tmelt,
)


# from functional.ffront.fbuiltins import exp #TODO: Requires math builtins

# TODO: Local constants What to do with these?
cp_v = 1850.0  # specific heat of water vapor at constant pressure (Landolt-Bornstein)
ci = 2108.0  # specific heat of ice

tol = 1e-3
maxiter = 10
zqwmin = 1e-20

# TODO: Docstrings will crash field_Operators
"""
Return latent heat of vaporization.

Computed as internal energy and taking into account Kirchoff's relations
"""


@field_operator
def _latent_heat_vaporization(
    t: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:

    return alv + (cp_v - clw) * (t - tmelt) - rv * t


"""Return saturation water vapour pressure."""


@field_operator
def _sat_pres_water(t: Field[[CellDim, KDim], float]) -> Field[[CellDim, KDim], float]:

    # DL: TODO swicth back once math-builtins available in GT4Py
    # return c1es * exp(c3les * (t - tmelt) / (t - c4les)) # noqa: E800
    return c1es * (c3les * (t - tmelt) / (t - c4les))


"""Return specific humidity at water saturation (with respect to flat surface)."""


@field_operator
def _qsat_rho(
    t: Field[[CellDim, KDim], float], rho: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:
    return _sat_pres_water(t) / (rho * rv * t)


"""
Return partial derivative of the specific humidity at water saturation.

Computed with respect to the temperature at constant total density.
"""


@field_operator
def _dqsatdT_rho(
    t: Field[[CellDim, KDim], float], zqsat: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:

    beta = c5les / (t - c4les) ** 2 - 1.0 / t
    return beta * zqsat


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

    # TODO: DL: Inline below??
    lwdocvd = _latent_heat_vaporization(t) / cvd

    # check, which points will still be subsaturated even after evaporating
    # all cloud water.For these gridpoints Newton iteration is not necessary.
    # TODO: to implement conditionals in gt4py, the 'where' builtin has to be used

    if qv + qc <= _qsat_rho(t, rho):
        # All the cloud water is evaporated, but (sub)saturation remains .
        # No iteration of temperature needed.
        qv = qv + qc
        qc = 0.0
        t = t - lwdocvd * qc
    else:
        # Save the starting value of temperature t
        twork = t

        # Storage variable for the "old" values in the below iteration.
        # Add nonesense increment to trigger the iteration.
        # TODO: DL: Make nicer with iteration functionality
        tworkold = twork + 10.0

        for _ in 1, maxiter:
            if abs(twork - tworkold) > tol:
                # Here we still have to iterate ...
                tworkold = twork

                # Helpers (TODO refactor and remove?)
                qwd = _qsat_rho(t, rho)
                dqwd = _dqsatdT_rho(twork, qwd)

                # Newton
                fT = twork - t + lwdocvd * (qwd - qv)
                dfT = 1.0 + lwdocvd * dqwd
                twork = twork - fT / dfT

        t = twork

        # The extrapolation of qsat from the second-last iteration step
        # is disregarded. This is typically done to exactly preserve the
        # internal energy. Omitting that introduces only a small error.
        qwa = _qsat_rho(t, rho)
        qc = max(qc + qv - qwa, zqwmin)
        qv = qwa
    return qv, qc, t


@program
def satad(
    qv: Field[[CellDim, KDim], float],
    qc: Field[[CellDim, KDim], float],
    t: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
):
    _satad(qv, qc, t, rho, out=(qv, qc, t))
