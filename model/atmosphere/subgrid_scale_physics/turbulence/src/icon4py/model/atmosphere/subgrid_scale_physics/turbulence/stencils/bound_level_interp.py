# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""'bound_level_interp': the mass-weighted interpolation of section 0) onto the half levels.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'bound_level_interp'
(:3290-3380 at icon commit 26d6b98cce, the commit that produced the reference capture), as
'turbdiff' section 0) calls it at :1084-1086 under the heading "Noetige Interpolationen auf
Nebenflaechen". The scientific commentary in that file is by Matthias Raschendorfer (DWD).

THREE STATEMENTS, and the third is not ICON's:

  1. the interpolation weight 'hlp', the thickness of the upper of the two main layers that meet
     at a half level over the sum of both;
  2. the seven interpolated profiles, over 'k_st=2 .. k_en=ke';
  3. THE ROW-0 PASS-THROUGH, which exists only because the port cannot interpolate in place.

STATEMENT 3 REPLACES FOUR HOST-SIDE COPIES, and it is the reason this merge is worth more than a
name. Four of the seven interpolations are IN PLACE in the Fortran -- 'rcld' and 'zaux(:,:,3..5)'
arrive holding the main-level values and leave holding the half-level ones on rows 1..ke-1 -- so
row 0 is the main-level value, for free, because nothing wrote it. GT4Py cannot read and write
one field across a vertical shift, so the port interpolates out of place and 'run_turbdiff' put
row 0 back with four '_copy_level' calls. As a statement over '(vertical_start - 1,
vertical_start)' they are one kernel inside the unit that owes them, and the granule loses four
host round trips per call.

The other three interpolations have no such row: the Exner factor and the density are read from
their own main-level storages ('epr', 'rhoh') rather than written in place, and the half-level
pressure's row 0 is untouched memory on both sides.

ONE VERTICAL PAIR, TWO RANGES, both arithmetic on it: 'k_st=2 .. k_en=ke' is
'(vertical_start, vertical_end)', and the pass-through is the single row above it.

THE INTERPOLATION WEIGHT IS KEPT AS AN OUTPUT even though it is a pure intermediate: 'hlp' at
'turbdiff-0-exit' holds it, so dropping it would cost a comparison against ICON for nothing.
Statement 2 reads it POINTWISE, which is ordinary dataflow within a program at any offset.

NO 'concat_where' AND NO SCAN, so this unit keeps its 'embedded' cross-check -- which matters
here more than anywhere else in 'turbdiff', because its gate is one of only three tolerant ones
and what makes that tolerance honest is that three backends agree with each other and differ
from nvhpc alone.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_half_level_interpolation_weight(
    layer_pressure_thickness: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """The weight of the LOWER of the two main levels that meet at a half level [-].

    It is the thickness of the UPPER layer over the sum of both, which is what makes the
    interpolated value lean towards the thicker layer: a half level that sits just below a thick
    layer and just above a thin one is nearly the thin layer's value.

    Only rows 1 to 'nlev - 1' are written -- the Fortran loop is 'DO k = ke, 2, -1' -- so run the
    program with 'vertical_start = 1', 'vertical_end = nlev'. There is nothing to write at the
    model top or at the surface: neither half level lies between two main levels.

    Args:
        layer_pressure_thickness: 'dp0', the pressure thickness of the main layers [Pa]

    Returns:
        'hlp' as section 0) leaves it, the interpolation weight [-]
    """
    thickness_above = layer_pressure_thickness(Koff[-1])
    return thickness_above / (thickness_above + layer_pressure_thickness)


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


@gtx.field_operator
def _pass_the_main_level_values_through(
    cloud_cover: fa.CellKField[wpfloat],
    dqsat_dt: fa.CellKField[wpfloat],
    buoyancy_factor_tet_l: fa.CellKField[wpfloat],
    buoyancy_factor_h2o_g: fa.CellKField[wpfloat],
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """The identity, on the one row the Fortran gets for free by interpolating in place.

    'bound_level_interp' writes rows 'k_st .. k_en' of the storage it was handed, and for these
    four that storage already held the main-level profile, so row 'k_st - 1' comes out as the
    main-level value. The port interpolates out of place, so it has to say it.
    """
    return (cloud_cover, dqsat_dt, buoyancy_factor_tet_l, buoyancy_factor_h2o_g)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def bound_level_interp(
    layer_pressure_thickness: fa.CellKField[wpfloat],
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
    """Interpolate the seven main-level profiles onto the half levels, in three statements.

    Args:
        layer_pressure_thickness: 'dp0' [Pa], main levels; the weight's only input.
        cloud_cover: 'rcld' on main levels [-].
        exner_factor: 'epr' on main levels [-].
        dqsat_dt: 'zaux(:,:,3)', dQs/dT on main levels [1/K].
        buoyancy_factor_tet_l: 'zaux(:,:,4)' on main levels.
        buoyancy_factor_h2o_g: 'zaux(:,:,5)' on main levels.
        pressure: 'prs' on main levels [Pa].
        air_density: 'rhoh' on main levels [kg/m3].
        interpolation_weight: Output, 'hlp' [-]; a pure intermediate that the exit savepoint
            nevertheless holds, so it is published.
        cloud_cover_on_half_levels: Output, 'rcld' [-]. Its row 0 is the main-level value and
            its surface row is what the second 'adjust_satur_equil' call left there.
        exner_factor_on_half_levels: Output, 'zaux(:,:,1)' [-].
        dqsat_dt_on_half_levels: Output, 'zaux(:,:,3)' [1/K].
        buoyancy_factor_tet_l_on_half_levels: Output, 'zaux(:,:,4)'.
        buoyancy_factor_h2o_g_on_half_levels: Output, 'zaux(:,:,5)'.
        pressure_on_half_levels: Output, 'zvari(:,:,0)' [Pa].
        air_density_on_half_levels: Output, 'rhon' [kg/m3].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First interpolated half level; 1, mirroring Fortran 'k_st=2'.
        vertical_end: End of the interpolated half levels; 'ke', mirroring 'k_en=ke'.
    """
    _compute_half_level_interpolation_weight(
        layer_pressure_thickness=layer_pressure_thickness,
        out=interpolation_weight,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
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
    _pass_the_main_level_values_through(
        cloud_cover=cloud_cover,
        dqsat_dt=dqsat_dt,
        buoyancy_factor_tet_l=buoyancy_factor_tet_l,
        buoyancy_factor_h2o_g=buoyancy_factor_h2o_g,
        out=(
            cloud_cover_on_half_levels,
            dqsat_dt_on_half_levels,
            buoyancy_factor_tet_l_on_half_levels,
            buoyancy_factor_h2o_g_on_half_levels,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start - 1, vertical_start),
        },
    )
