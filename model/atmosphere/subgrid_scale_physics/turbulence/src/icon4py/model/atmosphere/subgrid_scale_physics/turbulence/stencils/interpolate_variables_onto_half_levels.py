# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The seven main-level profiles 'turbdiff' section 0) needs on half levels.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section 0),
the 'pvar' list at :1067-1085 and the call of 'bound_level_interp' it feeds (the callee is
turb_utilities.f90:3232-3367, the '%bl' block at :3304-3313). Lines are as of icon commit
26d6b98cce, the commit that produced the reference capture. The scientific commentary in both
files is by Matthias Raschendorfer (DWD).

    blvar(i,k) = mlvar(i,k)*auxil(i,k) + mlvar(i,k-1)*(1 - auxil(i,k))

ONE PROGRAM FOR SEVEN QUANTITIES, because the Fortran is one call for seven quantities: the
routine takes a vector of source/target pointer pairs and the whole point of it is that they
share the precomputed weight. Splitting it would multiply the launches by seven for no gain in
clarity -- the formula is written once either way -- and the shared weight makes them a single
operation rather than seven that happen to look alike.

WHICH SEVEN, AND WHY NOT EIGHT. The list is assembled by the caller and depends on two switches:

    n=0  rcld       <- rcld       the cloud cover, interpolated IN PLACE
    n=1  zaux(1)    <- epr        the Exner factor
         zaux(2)    <- zaux(2)    'r_cpd', ONLY IF 'lcpfluc'
    n=2  zaux(3)    <- zaux(3)    'dQ_sat/dT', in place
    n=3  zaux(4)    <- zaux(4)    the buoyancy factor of 'tet_l', in place
    n=4  zaux(5)    <- zaux(5)    the buoyancy factor of 'h2o_g', in place
    n=5  prss       <- prs        the air pressure, ONLY IF 'lcircterm .OR. loutthcrc'
    n=6  rhon       <- rhoh       the air density

'lcpfluc' is frozen '.FALSE.' by 'TurbulenceConfig' ("fluctuations of the heat capacity of air
not considered"), so 'r_cpd' is not in the list and its half-level value stays the constant 1
that the main-level and surface calls wrote. 'lcircterm' is "pat_len > 0 .AND. ltkenst", both
true in every configuration this port targets, so the pressure IS in the list; a run with
'pat_len = 0' would leave 'zvari(:,:,0)' undefined and section 6) would not read it either.

THE FIVE IN-PLACE PAIRS ARE NOT A RECURRENCE, despite the aliasing and despite the '!$ACC LOOP
SEQ' the Fortran needs for it. The loop runs 'DO k = ke, 2, -1' and writes row 'k' from rows 'k'
and 'k-1'; row 'k-1' is written one iteration LATER, so every read sees the main-level value the
section put there. Source and target are separate fields here and the direction is irrelevant.

WHY THE WEIGHTED FORM AND NOT 'zbnd_val'. 'bound_level_interp' computes the same interpolation
two ways -- with a precomputed weight when 'auxil' is present, and as
'(val1*dep2 + val2*dep1)/(dep1 + dep2)' otherwise (turb_utilities.f90:3319-3328) -- and this call
site passes 'auxil'. The two differ in the last bits, so the port has to take the same branch.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_onto_half_levels(
    main_level_profile: fa.CellKField[wpfloat],
    interpolation_weight: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """One main-level profile at the half levels between its levels."""
    return main_level_profile * interpolation_weight + main_level_profile(Koff[-1]) * (
        wpfloat("1.0") - interpolation_weight
    )


@gtx.field_operator
def _interpolate_variables_onto_half_levels(
    cloud_cover: fa.CellKField[wpfloat],
    exner_factor: fa.CellKField[wpfloat],
    dqsat_dt: fa.CellKField[wpfloat],
    buoyancy_factor_tet_l: fa.CellKField[wpfloat],
    buoyancy_factor_h2o_g: fa.CellKField[wpfloat],
    pressure: fa.CellKField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    interpolation_weight: fa.CellKField[wpfloat],
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """All seven at once, in the Fortran's own order.

    Rows 1 to 'nlev - 1' only: run the program with 'vertical_start = 1',
    'vertical_end = nlev'. Every one of the seven storages keeps at the model top whatever it
    held, and at the surface row what the second 'adjust_satur_equil' call put there -- which is
    exactly why the Fortran can call this after that one and not before.

    Args:
        cloud_cover: 'rcld' on main levels, from 'adjust_satur_equil' [-]
        exner_factor: 'epr', ICON's own Exner factor on main levels [-]
        dqsat_dt: 'zaux(:,:,3)' on main levels [1/K]
        buoyancy_factor_tet_l: 'zaux(:,:,4)' on main levels [m/s2/K]
        buoyancy_factor_h2o_g: 'zaux(:,:,5)' on main levels [m/s2]
        pressure: 'prs', the air pressure on main levels [Pa]
        air_density: 'rhoh', ICON's own air density on main levels [kg/m3]
        interpolation_weight: 'hlp', from 'compute_half_level_interpolation_weight' [-]

    Returns:
        the same seven on half levels
    """
    return (
        _interpolate_onto_half_levels(cloud_cover, interpolation_weight),
        _interpolate_onto_half_levels(exner_factor, interpolation_weight),
        _interpolate_onto_half_levels(dqsat_dt, interpolation_weight),
        _interpolate_onto_half_levels(buoyancy_factor_tet_l, interpolation_weight),
        _interpolate_onto_half_levels(buoyancy_factor_h2o_g, interpolation_weight),
        _interpolate_onto_half_levels(pressure, interpolation_weight),
        _interpolate_onto_half_levels(air_density, interpolation_weight),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_variables_onto_half_levels(
    cloud_cover: fa.CellKField[wpfloat],
    exner_factor: fa.CellKField[wpfloat],
    dqsat_dt: fa.CellKField[wpfloat],
    buoyancy_factor_tet_l: fa.CellKField[wpfloat],
    buoyancy_factor_h2o_g: fa.CellKField[wpfloat],
    pressure: fa.CellKField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    interpolation_weight: fa.CellKField[wpfloat],
    cloud_cover_on_half_levels: fa.CellKField[wpfloat],
    exner_factor_on_half_levels: fa.CellKField[wpfloat],
    dqsat_dt_on_half_levels: fa.CellKField[wpfloat],
    buoyancy_factor_tet_l_on_half_levels: fa.CellKField[wpfloat],
    buoyancy_factor_h2o_g_on_half_levels: fa.CellKField[wpfloat],
    pressure_on_half_levels: fa.CellKField[wpfloat],
    air_density_on_half_levels: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_variables_onto_half_levels(
        cloud_cover=cloud_cover,
        exner_factor=exner_factor,
        dqsat_dt=dqsat_dt,
        buoyancy_factor_tet_l=buoyancy_factor_tet_l,
        buoyancy_factor_h2o_g=buoyancy_factor_h2o_g,
        pressure=pressure,
        air_density=air_density,
        interpolation_weight=interpolation_weight,
        out=(
            cloud_cover_on_half_levels,
            exner_factor_on_half_levels,
            dqsat_dt_on_half_levels,
            buoyancy_factor_tet_l_on_half_levels,
            buoyancy_factor_h2o_g_on_half_levels,
            pressure_on_half_levels,
            air_density_on_half_levels,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
