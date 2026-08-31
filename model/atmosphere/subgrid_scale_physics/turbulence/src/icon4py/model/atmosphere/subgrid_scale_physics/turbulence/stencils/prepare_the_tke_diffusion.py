# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Section 6) of 'turbdiff': everything the vertical TKE diffusion needs before it is solved.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', from the
section whose banner reads

    "6) Berechnung der Diffusionstendenz von q=SQRT(2*TKE) einschliesslich der q-Tendenz durch
        den Zirkulationsterm"
    -- "Calculation of the diffusion tendency of q = SQRT(2*TKE), including the q-tendency due
       to the circulation term"

(:2118-2310 at icon commit 26d6b98cce, the commit that produced the reference capture). The
scientific commentary in that file is by Matthias Raschendorfer (DWD).

ONE PROGRAM, FOUR STATEMENTS, TWO DOMAINS. Raschendorfer's own name for the first half of the
section is the name of this program: "Vorbereitung zur Bestimmung der zugehoerigen Incremente
von TKE=(q**2)/2" -- "Preparation for determining the corresponding increments of TKE = q**2/2"
(:2122-2124). The four statements are

  1. the pre-diffusion TKE profile 'sav_prof', half levels 1..ke1;
  2. the explicit diffusion momentum 'expl_mom', flux levels 2..ke1;
  3. the scaled CKE flux density 'frh', half levels 1..ke1;
  4. that flux interpolated onto the flux levels as 'frm', 2..ke1.

The flux levels start one row lower than the half levels, which is the whole of the difference
between the two domains: a flux level lies ABOVE the half level of the same index, so the
topmost half level has no flux level above it that carries a defined pair.

THE ORDER OF THE FIRST TWO STATEMENTS IS THE ONE THE FORTRAN FORBIDS, deliberately. In the
Fortran the TKE diffusion coefficient and the saved TKE profile share the storage 'zaux(:,:,2)',
so the loop that averages the coefficient onto the flux levels HAS to run before the one that
overwrites it. That is an aliasing constraint, not a data dependence: here the coefficient is an
intermediate inside statement 2, which reads 'len_scale' and 'tke' and never the saved profile.
Writing 'sav_prof' first and still reproducing ICON's 'expl_mom' bit for bit is what says so, and
'test_the_two_zaux_programs_do_not_constrain_each_others_order' is the assertion.

Statement 4 does read what statement 3 wrote -- that one is a genuine dependence, and its
position in the source is now what expresses it.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_saved_tke_profile(
    turbulent_velocity_scale: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Turbulent kinetic energy from the turbulent velocity scale, 'TKE = q**2 / 2'.

    THE SQUARE IS WRITTEN AS A PRODUCT, NOT AS '**2'. Fortran's integer-exponent '**' is a
    multiplication, while GT4Py lowers Python's 'x**2' to 'math.pow(x, 2)' and CUDA's 'pow'
    carries up to 2 ulp. See '_compute_mechanical_forcing', which is where that was measured.

    The half is applied to the square rather than to one factor, mirroring the Fortran's
    'z1d2*tke(i,k,ntur)**2' in which '**' binds tighter than '*'. (Both associations happen to
    give the same answer here, since multiplying by 0.5 is exact away from the subnormals, but
    the port mirrors the operation order rather than arguing about when it may not.)

    WHY THERE IS NO 'imode_tkediff == 1' BRANCH. The Fortran has one -- at 'imode_tkediff == 1'
    the diffusion is formulated in 'q' rather than in TKE, so the saved profile is 'q' itself and
    the discretisation momentum 'dicke' picks up an extra factor 'q' (:2187-2207).
    'imode_tkediff' is frozen at its compiled-in default 2 in this port ('turbulence.py',
    FROZEN_SWITCHES; port spec 4.3: not one of the 22 "which formulation" switches is set in any
    of the 641 configurations under 'icon/run/'), so only the TKE formulation is ported. Its
    signature in the reference data is that 'dicke' is byte-identical across this section, which
    'test_the_capture_diffuses_tke_and_not_q' asserts.
    """
    return wpfloat("0.5") * (turbulent_velocity_scale * turbulent_velocity_scale)


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

    From the two loops Raschendorfer heads "Diffusions-Koeffizienten auf NF:" -- "Diffusion
    coefficients at half levels" -- at :2141-2174.

    Density times diffusion coefficient over layer depth: the mass flux per unit gradient that
    the vertical diffusion of TKE transports across a flux level. The coefficient itself lives
    at half levels, so the value at the flux level between two of them is their arithmetic
    mean.

    The Fortran computes that coefficient into 'sav_prof' and then destroys it two loops later by
    storing the pre-diffusion TKE profile in the same slot, so the coefficient never reaches a
    savepoint. It is a genuine per-half-level intermediate, not an output, and it is therefore
    computed inside this operator rather than materialised in a field of its own.

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


@gtx.field_operator
def _compute_cke_flux_density(
    air_density: fa.CellKField[wpfloat],
    scalar_diffusion_coefficient: fa.CellKField[wpfloat],
    circulation_acceleration: fa.CellKField[wpfloat],
    mixing_length: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'rho_n * tkvh * a_circ * l', the length-scale-scaled CKE flux density.

    A plain product of four half-level fields, in the Fortran's order. From the block
    Raschendorfer heads "Aufnahme des Zirkulationstermes mit Interpolation auf HF:" -- "Taking up
    the circulation term, with interpolation onto main levels" -- at :2215-2217, first loop
    (:2221-2237), with his heading and trailing comment:

        ! Belegung von 'frh' mit der CKE-Flussdichte durch nicht-turbulente Zirkulationen, die
        !  durch thermische Inhomogenitaet an der Oberflaeche verursacht wird:
        frh(i,k) = rhon(i,k)*tkvh(i,k)*prss(i,k)*len_scale(i,k)   ! skalierte Flussdichte auf NF

        -- "Filling 'frh' with the flux density of circulation kinetic energy carried by the
           non-turbulent circulations that the thermal inhomogeneity of the surface causes"
           ... "scaled flux density at half levels"

    WHY THE FLUX IS SCALED BY THE LENGTH SCALE, from the note at :2231-2234:

        "'frh/len_scale' ist eine TKE-Flussdichte in [Kg/s3], deren Vertikalprofil im
         wesentlichen durch d_z(tet_v)**2 bestimmt ist, was zumindest in der Prandtl-Schicht
         prop. zu 1/len_scale ist. Die nachfolgende lineare Interpolation auf Hauptflaechen
         erfolgt daher mit 'frh'!"
        -- "'frh/len_scale' is a TKE flux density in [kg/s3] whose vertical profile is
           essentially determined by d_z(tet_v)**2, which -- at least within the Prandtl layer
           -- is proportional to 1/len_scale. The subsequent linear interpolation onto main
           levels is therefore performed on 'frh'!"

    In other words the factor 'len_scale' is not part of the physical flux; it is what makes
    the quantity smooth enough in the vertical that the linear interpolation of the next
    statement is legitimate, and that operator divides it out again.

    THE STORAGE IS REUSED AND SO IS THE MEANING. 'frh' held the thermal (buoyancy) forcing of
    the TKE equation from section 1b) up to and including section 5); this statement overwrites
    it with a flux density. Likewise 'circulation_acceleration' is the storage 'zvari(:,:,0)',
    which entered 'turbdiff' as the half-level air pressure ('prss => zvari(:,:,0)',
    turb_diffusion.f90:859) and which 'solve_turb_budgets' replaced in section 3) by the
    circulation acceleration proper, 'l_coh * fh2' (turb_utilities.f90:1728-1744). The Fortran
    name 'prss' at :2229 is therefore stale by two sections; it is the acceleration in [m/s2]
    that this product needs for its units to come out as [kg m/s3].

    The whole block is guarded by 'IF (lcircterm .OR. loutthcrc)', i.e. by 'pat_len > 0'
    (and 'ltkenst' for the first half). It is on in the reference capture, where
    'pat_len = 750 m'.
    """
    return air_density * scalar_diffusion_coefficient * circulation_acceleration * mixing_length


@gtx.field_operator
def _compute_cke_flux_at_main_levels(
    cke_flux_density: fa.CellKField[wpfloat],
    mixing_length: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Linear interpolation of the scaled flux onto the main level between two half levels.

    From the loop Raschendorfer heads "Interpolation der skalierten CKE-Flussdichte auf
    Hauptflaechen:" -- "Interpolation of the scaled CKE flux density onto main levels" --
    at :2239-2249. The Fortran is

        frm(i,k) = (frh(i,k) + frh(i,k-1)) / (len_scale(i,k) + len_scale(i,k-1))

    which is the mean of the two scaled fluxes divided by the mean of the two length scales --
    the two halves cancel, which is why neither appears. The result is the UNSCALED flux
    density [kg/s3] at the flux level, so this operator both interpolates and undoes the
    scaling that the previous statement applied.

    Section 8) divides this by 'expl_mom' to obtain the contribution of the circulation flux to
    the virtual TKE profile, so it carries the same staggering: index k is the flux level ABOVE
    half level k, i.e. main level k-1. See '_compute_explicit_tke_diffusion_momentum' for
    Raschendorfer's statement of that convention.

    THIS IS NOT A RECURRENCE. 'frh' is written by the preceding statement and never by this one;
    the Fortran writes into a different array, 'frm'. The Fortran loop is a plain
    '!$ACC LOOP GANG VECTOR COLLAPSE(2)'.

    THE STORAGE IS REUSED. 'frm' held the mechanical (shear) forcing of the TKE equation from
    section 1b) to section 5); this statement overwrites it. The two are not related, and the
    savepoint reader gives them separate accessors ('mech_forcing()' and
    'cke_flux_at_main_levels()') for that reason.
    """
    return (cke_flux_density + cke_flux_density(Koff[-1])) / (
        mixing_length + mixing_length(Koff[-1])
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def prepare_the_tke_diffusion(
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    mixing_length: fa.CellKField[wpfloat],
    air_density_at_main_levels: fa.CellKField[wpfloat],
    half_level_height: fa.CellKField[wpfloat],
    tke_diffusion_factor: wpfloat,
    air_density: fa.CellKField[wpfloat],
    scalar_diffusion_coefficient: fa.CellKField[wpfloat],
    circulation_acceleration: fa.CellKField[wpfloat],
    saved_tke_profile: fa.CellKField[wpfloat],
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    cke_flux_density: fa.CellKField[wpfloat],
    cke_flux_at_main_levels: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Prepare the TKE diffusion: the saved profile, the diffusion momentum and the CKE flux.

    THE TWO VERTICAL RANGES, and why they differ by exactly one row at the top. The half-level
    quantities 'sav_prof' and 'frh' run over 'DO k=2,ke1'; the flux-level quantities 'expl_mom'
    and 'frm' over 'DO k=3,ke1', so their statements start at 'vertical_start + 1'. Flux level k
    lies above half level k, so flux level 1 would be the main level above the topmost half
    level, which is outside the atmosphere. Neither range includes the model top: 'turbdiff' has
    no TKE-diffusion level there. The surface half level IS written by all four, unlike in most
    other sections.

    Args:
        turbulent_velocity_scale: 'tke(:,:,ntur)', 'q = SQRT(2*TKE)' [m/s], half levels, as
            sections 3) and 4) leave it.
        mixing_length: 'len_scale', the turbulent master length scale [m], half levels.
        air_density_at_main_levels: 'rhoh' [kg/m3], main levels; read one level up, at the main
            level the flux level is.
        half_level_height: 'hhl' [m] ('p_metrics%z_ifc'), nlev + 1 levels.
        tke_diffusion_factor: 'c_diff_llim', the low-limited length-scale factor of the TKE
            diffusion [-]; see '_tke_diffusion_coefficient'.
        air_density: 'rhon' [kg/m3], half levels, as section 0) interpolated it.
        scalar_diffusion_coefficient: 'tkvh' [m2/s], half levels, as section 4) limited it.
        circulation_acceleration: 'prss' = 'zvari(:,:,0)' [m/s2], half levels: the vertical
            acceleration of the near-surface thermal circulations, as 'solve_turb_budgets' left
            it in section 3). NOT the pressure the Fortran name suggests.
        saved_tke_profile: Output, 'sav_prof' = 'zaux(:,:,2)', turbulent kinetic energy
            [m2/s2], half levels.
        explicit_diffusion_momentum: Output, 'expl_mom' = 'zaux(:,:,3)' [kg/m2/s], at the flux
            levels, indexed so that flux level k lies above half level k.
        cke_flux_density: Output, 'frh' [kg m/s3], half levels; read by the last statement.
        cke_flux_at_main_levels: Output, 'frm' [kg/s3], at the flux levels.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k=2'. The flux-level statements
            start one row lower.
        vertical_end: End of both ranges; 'ke1', mirroring Fortran 'k=...,ke1'.
    """
    _compute_saved_tke_profile(
        turbulent_velocity_scale=turbulent_velocity_scale,
        out=saved_tke_profile,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_explicit_tke_diffusion_momentum(
        mixing_length=mixing_length,
        turbulent_velocity_scale=turbulent_velocity_scale,
        air_density_at_main_levels=air_density_at_main_levels,
        half_level_height=half_level_height,
        tke_diffusion_factor=tke_diffusion_factor,
        out=explicit_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start + 1, vertical_end),
        },
    )
    _compute_cke_flux_density(
        air_density=air_density,
        scalar_diffusion_coefficient=scalar_diffusion_coefficient,
        circulation_acceleration=circulation_acceleration,
        mixing_length=mixing_length,
        out=cke_flux_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_cke_flux_at_main_levels(
        cke_flux_density=cke_flux_density,
        mixing_length=mixing_length,
        out=cke_flux_at_main_levels,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start + 1, vertical_end),
        },
    )
