# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Vertical acceleration that expresses the raw "circulation term".

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'solve_turb_budgets', from the block headed "Calculating vertical acceleration (CKE-gradient)
used for the raw 'circulation term'" (:1728-1748 at icon commit 26d6b98cce). Called from
'turbdiff' section 3) (turb_diffusion.f90:1795-1914). The scientific commentary is by Matthias
Raschendorfer (DWD).

WHAT IT MODELS. Near-surface thermal inhomogeneity -- a land-use pattern, a partly cloudy sky --
drives circulations that are not resolved and are not turbulence either. The scheme accounts for
them as an additional TKE source, and this is its accelerating term: a vertical coherence length
'l_coh' times the thermal forcing 'fh2'. The result is a gradient of circulation kinetic energy
per unit mass, in [m/s2], written into the 'zvari' component 0 that held the half-level pressure
until now.

THE COHERENCE LENGTH is the larger of the pure land-use pattern scale 'l_pat' and a
cloud-cover-weighted geometric mean of the master length scale and the horizontal grid spacing.
Its cloud weight, 'fakt = 1 - 2*|rcld - 1/2|', peaks at half cloud cover and vanishes at a clear
or a fully overcast sky, which is where broken-cloud circulations are strongest and weakest. It
is then scaled by the dimensionless virtual potential temperature gradient, clipped to the unit
interval and signed by it.

THIS BLOCK READS 'rcld' AS THE CLOUD COVER, and the block that computes the standard deviation
of the local super-saturation overwrites 'rcld' with that. In the Fortran the ordering is what
keeps them apart; here they are separate fields and the ordering is free. Do not restore the
aliasing.
"""

import gt4py.next as gtx
from gt4py.next import abs, maximum, minimum, sqrt  # noqa: A004 [builtin-shadowing]

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_circulation_acceleration(
    cloud_cover: fa.CellKField[wpfloat],
    master_length_scale: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    half_level_pressure: fa.CellKField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    pattern_length_scale: fa.CellField[wpfloat],
    horizontal_grid_scale: fa.CellField[wpfloat],
    gravitational_acceleration: wpfloat,
) -> fa.CellKField[wpfloat]:
    """The CKE gradient 'l_coh*fh2' [m/s2].

    The Fortran is four statements (turb_utilities.f90:1738-1745), the middle two reusing the
    name 'fakt' for two unrelated quantities:

        fakt  = z1-z2*ABS(rcld(i,k)-z1d2)                  ! coherence factor by cloud cover
        l_coh = MAX( l_pat(i), SQRT(fakt*tls(i,k)*l_hori(i)) )
        fakt  = fh2(i,k)*grd(i,k,0)/(dens(i,k)*grav2)      ! = Rd/g*exnr*grad(tet_v)
        l_coh = l_coh*SIGN(z1,fakt)*MIN( ABS(fakt), z1 )
        grd(i,k,0) = l_coh*fh2(i,k)

    with 'grd(:,:,0)' the half-level pressure on input and the acceleration on output, and
    'grav2 = grav**2' precomputed in the caller (:1310).

    THE SIGNED CLIP. 'SIGN(1,x)*MIN(|x|,1)' is the clamp of 'x' to [-1, 1], and that is how it
    is written here: 'maximum(-1, minimum(x, 1))'. The two forms agree bit for bit, signed zeros
    included -- multiplying by an exact +-1 is exact, so the rewrite only reorders exact
    operations -- and the clamp additionally avoids the one place where a naive translation
    would NOT agree. Fortran's 'SIGN(1.0, -0.0)' is -1.0, while a 'where(x < 0, -1, +1)' gives
    +1.0, because '-0.0 < 0' is false; the two then differ in the sign of a zero result. No
    point of the reference capture has 'x == 0' at all (asserted by
    'test_the_coherence_scaling_is_never_handed_an_exact_zero'), so this is a gap the data
    cannot close and the clamp closes structurally.

    Args:
        cloud_cover: 'rcld' as 'adjust_satur_equil' and 'bound_level_interp' leave it, the
            saturation fraction on half levels [-].
        master_length_scale: 'tls' ('len_scale'), turbulent master length scale [m].
        thermal_forcing: 'fh2' ('frh'), squared frequency of the thermal forcing [1/s2].
        half_level_pressure: 'grd(:,:,0)' ('zvari(:,:,0)'), the half-level pressure [Pa].
        air_density: 'dens' ('rhon'), air density on half levels [kg/m3].
        pattern_length_scale: 'l_pat', effective length scale of the near-surface circulation
            patterns [m], one value per column.
        horizontal_grid_scale: 'l_hori', horizontal grid spacing [m], one value per column.
        gravitational_acceleration: 'grav' [m/s2]; squared here, as 'grav2' is in the Fortran.

    Returns:
        The circulation acceleration 'grd(:,:,0)' [m/s2].
    """
    coherence_factor = wpfloat("1.0") - wpfloat("2.0") * abs(cloud_cover - wpfloat("0.5"))
    coherence_length = maximum(
        pattern_length_scale,
        sqrt(coherence_factor * master_length_scale * horizontal_grid_scale),
    )
    virtual_temperature_gradient = (
        thermal_forcing
        * half_level_pressure
        / (air_density * (gravitational_acceleration * gravitational_acceleration))
    )
    coherence_length = coherence_length * maximum(
        -wpfloat("1.0"), minimum(virtual_temperature_gradient, wpfloat("1.0"))
    )
    return coherence_length * thermal_forcing


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_circulation_acceleration(
    cloud_cover: fa.CellKField[wpfloat],
    master_length_scale: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    half_level_pressure: fa.CellKField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    pattern_length_scale: fa.CellField[wpfloat],
    horizontal_grid_scale: fa.CellField[wpfloat],
    gravitational_acceleration: wpfloat,
    circulation_acceleration: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute the CKE gradient on all boundary levels, the extra lowermost one included.

    The vertical domain is the Fortran 'DO k=k_st,k_sf', "for all boundary levels including the
    extra lowermost boundary": 'k_st = 2' and 'k_sf = ke1' (turb_diffusion.f90:1814), which is
    half levels 1 to 'ke' zero-based, one level deeper than every other stencil of section 3).
    The surface row matters -- it is where the near-surface circulations are anchored.

    Runs only on the last iteration step of 'solve_turb_budgets' and only under 'lcircterm',
    which is 'pat_len > 0 .AND. ltkenst' (turb_diffusion.f90:944, :959). With 'it_end = 1' the
    first condition is automatic; the second is a configuration question, and where it does not
    hold this program must not run, since nothing else restores the half-level pressure it
    overwrites.

    Args:
        cloud_cover: 'rcld' [-], half levels.
        master_length_scale: 'tls' [m].
        thermal_forcing: 'fh2' [1/s2].
        half_level_pressure: 'zvari(:,:,0)' [Pa].
        air_density: 'rhon' [kg/m3].
        pattern_length_scale: 'l_pat' [m], one value per column.
        horizontal_grid_scale: 'l_hori' [m], one value per column.
        gravitational_acceleration: 'grav' [m/s2].
        circulation_acceleration: Output, 'zvari(:,:,0)' [m/s2].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k_st = 2'.
        vertical_end: End of the half levels; 'ke1', mirroring 'k_sf = ke1'.
    """
    _compute_circulation_acceleration(
        cloud_cover=cloud_cover,
        master_length_scale=master_length_scale,
        thermal_forcing=thermal_forcing,
        half_level_pressure=half_level_pressure,
        air_density=air_density,
        pattern_length_scale=pattern_length_scale,
        horizontal_grid_scale=horizontal_grid_scale,
        gravitational_acceleration=gravitational_acceleration,
        out=circulation_acceleration,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
