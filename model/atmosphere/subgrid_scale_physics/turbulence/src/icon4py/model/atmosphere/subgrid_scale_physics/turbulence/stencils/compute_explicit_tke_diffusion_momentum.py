# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit part of the vertical diffusion momentum of the TKE equation.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
6) (:2118-2310 at icon commit 26d6b98cce), from the two loops Raschendorfer heads

    "Diffusions-Koeffizienten auf NF:"
    -- "Diffusion coefficients at half levels"

at :2141-2174. The scientific commentary in that file is by Matthias Raschendorfer (DWD).

The Fortran computes the diffusion coefficient into 'sav_prof' and then destroys it two loops
later by storing the pre-diffusion TKE profile in the same slot, so the coefficient never
reaches a savepoint. It is a genuine per-half-level intermediate, not an output, and it is
therefore computed inside the field operator here rather than materialised in a field of its
own; see 'compute_saved_tke_profile' for what ends up in that storage.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _tke_diffusion_coefficient(
    mixing_length: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    tke_diffusion_factor: wpfloat,
) -> fa.CellKField[wpfloat]:
    """Diffusion coefficient for TKE at a half level, 'c_diff * l * q' [m2/s].

    The same closure as the diffusion coefficients for momentum and heat -- a length scale
    times a velocity scale -- with the length-scale factor 'c_diff' of the TKE equation, which
    'mo_turbdiff_config.f90:239-240' documents as covering the turbulent pressure transport as
    well as the turbulent transport proper.

    Raschendorfer's alternative, commented out beside the assignment at :2148-2153, is to reuse
    the scalar diffusion coefficient instead:

        "test: TKE-Diffusion mit Stab.fnkt. fuer Skalare"      ! sav_prof(i,k)=c_diff_llim*tkvh(i,k)
        -- "test: TKE diffusion with the stability function for scalars"

    Called twice by the operator below rather than once on a shifted result, so that both
    half levels evaluate the identical expression.
    """
    return tke_diffusion_factor * mixing_length * turbulent_velocity_scale


@gtx.field_operator
def _compute_explicit_tke_diffusion_momentum(
    mixing_length: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    air_density_at_main_levels: fa.CellKField[wpfloat],
    half_level_height: fa.CellKField[wpfloat],
    tke_diffusion_factor: wpfloat,
) -> fa.CellKField[wpfloat]:
    """The explicit diffusion momentum 'expl_mom' [kg/m2/s] of the TKE equation.

    Density times diffusion coefficient over layer depth: the mass flux per unit gradient that
    the vertical diffusion of TKE transports across a flux level. The coefficient itself lives
    at half levels, so the value at the flux level between two of them is their arithmetic
    mean.

    THE INDEX SHIFT IS THE FORTRAN'S, and it is not the usual one. Raschendorfer's note at
    :2170-2173:

        "'expl_mom' bezieht sich auf HF, also die Fluss-Niveaus fuer die TKE (bzw. q-)-Diffusion.
         Wegen der spaeteren Nutzung der SUBs 'prep_impl_vert_diff' und 'calc_impl_vert_diff'
         muss ein Fluss-Niveau (hier HF) ueber dem Variabl.-Niveau (hier NF) mit gleichem Index
         liegen."
        -- "'expl_mom' refers to main levels, i.e. the flux levels of the TKE (or q) diffusion.
           Because of the later use of SUBs 'prep_impl_vert_diff' and 'calc_impl_vert_diff', a
           flux level (here a main level) must lie ABOVE the variable level (here a half level)
           carrying the same index."

    So 'expl_mom(k)' sits at the main level between half levels k-1 and k, which is why every
    input on the numerator is taken at k-1 and k, and why the layer depth is
    'hhl(k-1) - hhl(k)' -- the depth of main layer k-1, the one this flux level is.

    THIS IS NOT A RECURRENCE. It reads the diffusion coefficient one half level up, but that
    coefficient is a function of inputs only; nothing here reads a value this operator wrote.
    The Fortran's own loop is 'DO k=3,ke1' inside a plain '!$ACC LOOP GANG VECTOR COLLAPSE(2)',
    i.e. Raschendorfer runs it fully parallel too.
    """
    coefficient_above = _tke_diffusion_coefficient(
        mixing_length(Koff[-1]), turbulent_velocity_scale(Koff[-1]), tke_diffusion_factor
    )
    coefficient_here = _tke_diffusion_coefficient(
        mixing_length, turbulent_velocity_scale, tke_diffusion_factor
    )
    return (
        air_density_at_main_levels(Koff[-1])
        * wpfloat("0.5")
        * (coefficient_above + coefficient_here)
        / (half_level_height(Koff[-1]) - half_level_height)
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_explicit_tke_diffusion_momentum(
    mixing_length: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    air_density_at_main_levels: fa.CellKField[wpfloat],
    half_level_height: fa.CellKField[wpfloat],
    tke_diffusion_factor: wpfloat,
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'expl_mom' at the flux levels of the TKE diffusion (turb_diffusion.f90:2142-2175).

    THE FACTOR IS THE LOW-LIMITED 'c_diff', NOT 'c_diff' ITSELF (:2135-2139):

        IF (lcircterm) THEN               ! raw "circulation term" enters the TKE equations
           c_diff_llim = MAX(tdc%epsi, tdc%c_diff)
        ELSE
           c_diff_llim = tdc%c_diff
        END IF

    which the caller is expected to have evaluated -- it is one scalar per run, not per column.
    The limit exists because section 8) divides the circulation flux by 'expl_mom' to build the
    virtual TKE profile, and 'c_diff = 0.0' together with an active circulation term would make
    that a division by zero (icon commit 571b0c9e42, "Enabling c_diff=0 in case of lcircterm=T").
    It is undone where it would otherwise change the physics: section 8) multiplies by
    'fakt = c_diff / c_diff_llim' so that pure TKE diffusion always acts with the unlimited
    value. In the reference capture 'c_diff = 0.2' and 'epsi = 1e-6', so the limit does not bind
    and 'test_the_c_diff_lower_limit_does_not_bind_in_this_capture' says so.

    Args:
        mixing_length: 'len_scale', the turbulent master length scale [m], half levels.
        turbulent_velocity_scale: 'tke(:,:,ntur)', 'q = SQRT(2*TKE)' [m/s], half levels.
        air_density_at_main_levels: 'rhoh' [kg/m3], main levels; read one level up, at the main
            level this flux level is.
        half_level_height: 'hhl' [m] ('p_metrics%z_ifc'), nlev + 1 levels.
        tke_diffusion_factor: 'c_diff_llim', the low-limited length-scale factor of the TKE
            diffusion [-]; see above.
        explicit_diffusion_momentum: Output, 'expl_mom' = 'zaux(:,:,3)' [kg/m2/s], at the flux
            levels, indexed so that flux level k lies above half level k.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First flux level; 2, mirroring Fortran 'k=3'. Rows 0 and 1 are not
            written: flux level 1 would be the main level above the first half level, which is
            outside the atmosphere.
        vertical_end: End of the flux levels; 'ke1', mirroring Fortran 'k=...,ke1'.
    """
    _compute_explicit_tke_diffusion_momentum(
        mixing_length=mixing_length,
        turbulent_velocity_scale=turbulent_velocity_scale,
        air_density_at_main_levels=air_density_at_main_levels,
        half_level_height=half_level_height,
        tke_diffusion_factor=tke_diffusion_factor,
        out=explicit_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
