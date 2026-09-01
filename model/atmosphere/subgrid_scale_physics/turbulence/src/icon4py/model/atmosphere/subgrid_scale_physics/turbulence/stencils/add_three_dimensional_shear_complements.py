# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Section 2a) of 'turbdiff': the three-dimensional and non-turbulent shear complements.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
2a) ("Adding 3D-complements of mechanical shear-forcing by the mean flow and all shear-forcing of
the non-turbulent sub-grid flow", :1323-1623 at icon commit 26d6b98cce, the commit that produced
the reference capture). The scientific commentary in that file is by Matthias Raschendorfer
(DWD); the two blocks banner-marked '!GZ: For tuning.' are Guenther Zaengl's.

The section takes the single-column mechanical forcing section 1b) left in 'frm' and adds
everything the mean flow and the non-turbulent sub-grid flow contribute to it. This program is
everything the section forms BEFORE those additions -- six statements, in the Fortran's order:

    :1338  frm(i,k) = MAX( (zvari(i,k,u_m)+dwdx(i,k))**2 + ..., fc_min(i) )   ! and += hdef2
    :1406  xri(i,k) = EXP( z2d3*LOG( MAX(1e-6,frm) / MAX(1e-5,frh) ) )        ! 1/Ri**(2/3)
    :1424  layr(i)  = a_hshr*akt/2 * l_hori(i)
    :1440  hor_scale(i,k) = layr(i)*MIN(5,MAX(0.01,x4*xri))/MAX(1,0.2*tke)
    :1509  hlp(i,k) = hlp(i,k)**3/hor_scale(i,k)                              ! -> tket_hshr
    :1560  hlp(i,k) = ut_sso(i,k)*u(i,k)+vt_sso(i,k)*v(i,k)                   ! MAIN levels

Each of the six was a '@gtx.program' of its own until the stencil merge; they are six statements
of one program now, which is what a reader of the Fortran expects to find. The statements are
otherwise unchanged -- the same field operators, the same domains, the same order, the same
outputs -- so nothing here is a numerical change.

TWO VERTICAL RANGES, BOTH ARITHMETIC ON THE ONE PAIR THE GRANULE BINDS. Five statements run over
'DO k=2,kem' with 'kem = ke', which is '(vertical_start, vertical_end)'; the SSO wake production
alone runs over 'DO k=1,kem' on MAIN levels, one row higher, which is 'vertical_start - 1'. The
length scale 'layr' has no vertical axis at all.

THE TOTAL FORCING IS NOT HERE, AND THE MERGE PLAN EXPECTED IT TO BE. Section 2a) finishes by
adding the separated shear and the SSO production to 'frm' (:1534, :1587-:1592), and 'imode_tkesso'
decides whether the second addition carries the Richardson reduction. Those are two programs,
'compute_total_mechanical_forcing' and '…_without_richardson_reduction', and they stay two: only
one of them may write 'frm', so folding the selected one in here would need a way to switch a
statement off, and the only such device is an EMPTY VERTICAL DOMAIN -- which is NOT a no-op on
the 'embedded' backend when the statement reads through a shift, and the total forcing reads
'hlp' and 'dp0' at 'Koff[-1]'. Measured 2026-09-01, '.scratch/merge3/toy_empty_embedded.py':

    embedded   pointwise, EMPTY (nlev, nlev)   ok, 0 rows written
    embedded   shifted,   EMPTY (nlev, nlev)   RAISED IndexOutOfBounds

gt4py normalises an empty 'UnitRange' to '(0, 0)' and then bounds-checks it against the SHIFTED
operand's domain, which starts at row 1 -- so the check fails for a statement that would have
done nothing. Step 7 of the merge plan measured the device on the four COMPILED backends only,
where it is sound; 'calc_impl_vert_diff' relies on it and is never run on 'embedded', so nothing
there is broken. Section 2a) is a section whose datatests DO run on 'embedded', which is how this
was found. Hence six statements and not seven, and section 2a) is three programs and not two.

WHY THE MEAN SHEAR IS STILL A FIELD. It is the Fortran's 'ftm' (:1382, "save traditional (pure
mean) shear") and it is read TWICE -- by the Richardson factor here and by the total forcing
afterwards -- so it cannot become an expression without evaluating it twice and rounding it
twice. That is the one place the merge plan expected a field to disappear and it does not.

THE THREE ALIASING RULES, which this program obeys and the next merge will meet again
(measured in 'solve_turb_budgets', whose docstring carries the five-variant table):

  * reading a parameter an EARLIER statement wrote is ordinary dataflow and is correct at any
    offset. Three statements here do it, and 'compute_total_mechanical_forcing' -- which runs
    next, on this program's outputs -- reads 'sso_wake_energy_production' at 'Koff[-1]';
  * writing a parameter the SAME statement reads is correct only pointwise -- DaCe silently
    drops the statement otherwise. No statement here does it: the Fortran accumulates into 'frm'
    three times and the port keeps the mean-shear part in a field of its own so that it does not;
  * an aliasing GT4Py cannot see -- one field bound to two parameters -- is correct only if the
    reader comes first. The caller binds seven distinct fields here, so there is none.

WHAT THIS SECTION WRITES, and what the capture does and does not cover, is documented in
'tests/turbulence/integration_tests/test_turbdiff_section_2a.py'; five namelist selectors decide
what runs and not one of them is serialized, so each is established from the data by a test of
its own.
"""

import gt4py.next as gtx
from gt4py.next import exp, log, maximum, minimum, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_three_dimensional_shear_forcing(
    vertical_gradient_u: fa.CellKField[wpfloat],
    vertical_gradient_v: fa.CellKField[wpfloat],
    dwdx: fa.CellKField[wpfloat],
    dwdy: fa.CellKField[wpfloat],
    horizontal_divergence: fa.CellKField[wpfloat],
    horizontal_deformation_square: fa.CellKField[wpfloat],
    min_forcing: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """The squared shear of the mean flow, floored by 'fc_min' and extended by the deformation.

    Section 1b) formed the single-column part of this, 'MAX(du/dz**2 + dv/dz**2, fc_min)'. Two
    corrections turn it into the three-dimensional one, and the Fortran's own comment block at
    :1345-1355 says what they are:

      * the vertical wind contributes to the horizontal-momentum shear, so its horizontal
        derivatives are added to the vertical derivatives of the horizontal wind, component by
        component. Incompressibility ('vel_div = hdiv + dw/dz = 0', :1351) turns the remaining
        diagonal term into '3 * hdiv**2';
      * 'hdef2', the squared horizontal deformation
        '(d1v2+d2v1)**2 + (d1v1-d2v2)**2' (:1345), is the purely horizontal shear.

    The floor is applied to the first group only, before 'hdef2' is added -- which is what the
    two separate Fortran loops encode and is the reason they are reproduced in this order and
    not fused into a single 'MAX'.

    ONLY 'itype_sher = 2' IS IMPLEMENTED. The Fortran selects among three formulations
    (:1353-1355): 0 is the single-column vertical shear alone, which section 1b) already
    computed; 1 adds 'hdef2'; 2 adds the vertical-wind terms as well. The reference capture ran
    2 -- measured from the data by 'test_the_capture_runs_the_full_three_dimensional_shear', not
    read off a namelist -- so 0 and 1 have no oracle here and are not written out.

    WHAT THIS OUTPUT IS, AND WHY IT IS NOT 'frm'. The section adds three further contributions to
    the same storage before its exit savepoint, so this value never reaches a savepoint of its
    own. It is not a nameless intermediate: it is exactly what Raschendorfer saves into 'ftm' at
    :1382, "save traditional (pure mean) shear", in the configurations that need the
    scale-interaction terms separable ('lssintact .OR. loutbms'). Neither holds here, so 'ftm' is
    untouched and the port carries the mean shear in a field of its own rather than accumulating
    into 'frm' in place -- which is also what keeps this section free of a statement that writes
    the parameter it reads. Its oracle is indirect but tight: 'xri' is a strictly monotone
    function of it and is compared bit for bit.

    THE FOUR 'vp' FIELDS. 'hdef2', 'hdiv', 'dwdx' and 'dwdy' are declared 'REAL(KIND=vp)'
    (turb_diffusion.f90:612-618) and come from the dycore's diffusion; this section is their only
    consumer in the whole of 'turbdiff'. They are typed 'wpfloat' here, as the savepoint reader
    returns them, because the default double build makes 'vp' equal to 'wp'. A mixed-precision
    build would need an 'astype' at the call site.

    THE SQUARES ARE WRITTEN AS PRODUCTS. Fortran's 'x**2' with an integer literal exponent is a
    multiplication, while GT4Py lowers Python's 'x**2' to 'math.pow'; CUDA's 'pow' carries up to
    2 ulp, so on a GPU backend the two are different numbers. See the docstring of
    '_compute_mechanical_forcing', where this was measured.
    """
    shear_u = vertical_gradient_u + dwdx
    shear_v = vertical_gradient_v + dwdy
    single_column_and_vertical_wind_shear = maximum(
        shear_u * shear_u
        + shear_v * shear_v
        + wpfloat("3.0") * (horizontal_divergence * horizontal_divergence),
        min_forcing,
    )
    return single_column_and_vertical_wind_shear + horizontal_deformation_square


@gtx.field_operator
def _compute_inverse_richardson_number_factor(
    mean_shear_forcing: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'EXP(2/3 * LOG(frm/frh))' with both arguments floored away from zero.

    'z2d3' is 'z2/z3' (turb_diffusion.f90:270), i.e. the double-precision quotient of the two
    integers and not a decimal literal, which is what '2.0/3.0' reproduces exactly.

    The gradient Richardson number is 'Ri = N**2 / S**2', the thermal forcing over the
    mechanical one, so 'frm/frh' is '1/Ri' and two thirds is the empirical exponent the tuning
    uses. The two floors are what make the quotient defined: 'frh' is a buoyancy forcing and is
    negative throughout the unstable boundary layer, and 'LOG' of a negative number is not a
    number. They are not symmetric -- 1e-6 on the shear, 1e-5 on the buoyancy -- so the neutral
    limit is 'xri = 0.1**(2/3)', a small number rather than one.

    'xri' is declared 'xri(nvec,ke)' (turb_diffusion.f90:802), one row shorter than the half
    levels, because the surface row is never needed; its values are half-level quantities all the
    same. The port keeps the Fortran's shape.

    THIS IS THE SECTION'S ONLY TRANSCENDENTAL AND THE ONLY THING IN IT THAT IS NOT BIT-EXACT.
    'EXP(LOG())' is evaluated by the target's libm and the reference was produced by nvhpc's.
    Every other quantity of the section reproduces ICON's bits exactly when it is handed ICON's
    own 'xri' -- asserted by 'test_the_section_is_bit_exact_when_the_transcendental_comes_from_
    icon' -- so the merged program's tolerant gate is this one rounding and nothing else. Writing
    it as a 'power' would not help: the same libm decides, and it would change which routine
    rounds. Its second consumer is section 4), for the Richardson-dependent minimum diffusion
    coefficients.
    """
    return exp(
        (wpfloat("2.0") / wpfloat("3.0"))
        * log(
            maximum(wpfloat("1.0e-6"), mean_shear_forcing)
            / maximum(wpfloat("1.0e-5"), thermal_forcing)
        )
    )


@gtx.field_operator
def _compute_uncorrected_horizontal_shear_length_scale(
    horizontal_mesh_size: fa.CellField[wpfloat],
    horizontal_shear_length_factor: wpfloat,
    karman_constant: wpfloat,
) -> fa.CellField[wpfloat]:
    """'a_hshr * akt / 2 * l_hori'.

    The parenthesisation follows the Fortran, which forms the scalar factor 'wert' once outside
    the loop and multiplies the field by it. Grouping the three scalars first is what keeps the
    result independent of the mesh size's own rounding.

    'l_hori' is the horizontal mesh size, so this is the mesh size scaled by the von Karman
    constant and the tuning factor 'a_hshr': the size of the largest eddy the separated
    horizontal shear mode can hold. ICON fills every entry of 'l_hori' with the single scalar
    'phy_params%mean_charlen', so it is constant over a domain, but it is a field.

    Nothing downstream of section 2a) reads 'layr' -- it is a Fortran scratch vector reused all
    over 'turbdiff' -- so it survives as an output only because the capture serializes it, which
    makes 'a_hshr' recoverable from the data. The reference run used 'a_hshr = 2.0' and not the
    compiled-in 1.0; 'test_the_capture_used_a_horizontal_shear_factor_of_two' recovers it.
    """
    return (
        horizontal_shear_length_factor * karman_constant * wpfloat("0.5")
    ) * horizontal_mesh_size


@gtx.field_operator
def _compute_effective_horizontal_shear_length_scale(
    uncorrected_horizontal_shear_length_scale: fa.CellField[wpfloat],
    half_level_height: fa.CellKField[wpfloat],
    surface_height: fa.CellField[wpfloat],
    inverse_richardson_number_factor: fa.CellKField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'layr' corrected by the height above ground, the stability and the turbulent velocity.

    'x4i**2' is written as a product: Fortran's integer-literal power is a multiplication while
    GT4Py's '**' becomes 'math.pow', which CUDA evaluates to within 2 ulp rather than exactly.
    See '_compute_mechanical_forcing' for where that was measured.

    Three corrections are applied to 'layr', in this order:

      * 'x4', a smoothstep in the height above ground. '0.5e-3 * dz' is 1 at 2000 m, so 'x4i'
        ramps from 0 at the surface to 1 at 2 km and is clipped there, and '(3 - 2*x4i)*x4i**2'
        is the cubic Hermite step that is flat at both ends. The separated horizontal shear mode
        is suppressed inside the boundary layer, where the vertical shear already accounts for
        it. Zaengl's own comment: "Factor for variable 3D horizontal-vertical length scale
        proportional to 1/SQRT(Ri), decreasing to zero in the lowest two kilometer above ground,
        from ICON 180206".
      * 'xri', the stability factor above. The product is clipped to [0.01, 5], so the correction
        spans not quite three decades and is neither zero nor unbounded however extreme the
        stratification.
      * a division by the turbulent velocity where that exceeds 5 m/s ('MAX(1, 0.2*q)'), which
        shrinks the scale in already strongly turbulent air.

    'imode_shshear' SELECTS THIS FORM. At any other value the Fortran skips all of it and takes
    'hor_scale(i,k) = layr(i)' unchanged (:1451-1463). The capture ran 'imode_shshear = 2' --
    established from the data, since the switch is not serialized, by
    'test_the_capture_corrects_the_shear_length_scale_by_the_richardson_number' -- so the plain
    branch has no oracle here and is not ported. 'TurbulenceConfig' freezes the switch at 2.
    """
    height_above_ground = minimum(
        wpfloat("1.0"), wpfloat("0.5e-3") * (half_level_height - surface_height)
    )
    low_level_reduction = (wpfloat("3.0") - wpfloat("2.0") * height_above_ground) * (
        height_above_ground * height_above_ground
    )
    return (
        uncorrected_horizontal_shear_length_scale
        * minimum(
            wpfloat("5.0"),
            maximum(wpfloat("0.01"), low_level_reduction * inverse_richardson_number_factor),
        )
        / maximum(wpfloat("1.0"), wpfloat("0.2") * turbulent_velocity_scale)
    )


@gtx.field_operator
def _compute_separated_horizontal_shear_tke_source(
    effective_horizontal_shear_length_scale: fa.CellKField[wpfloat],
    horizontal_divergence: fa.CellKField[wpfloat],
    horizontal_deformation_square: fa.CellKField[wpfloat],
    neutral_momentum_stability_function: wpfloat,
) -> fa.CellKField[wpfloat]:
    """'(hor_scale*(SQRT((fakt*hdiv)**2 + hdef2) - fakt*hdiv))**3 / hor_scale'.

    BOTH POWERS ARE WRITTEN AS PRODUCTS. Fortran's 'x**2' and 'x**3' with integer literal
    exponents are multiplications; GT4Py's '**' is 'math.pow', and CUDA's 'pow' carries up to
    2 ulp. Measured in '_compute_mechanical_forcing'. Cancelling the cube against the division
    algebraically -- 'hor_scale**2 * (...)**3' -- would be a third different rounding again, so
    the Fortran's two steps are kept as two steps.

    'fakt' is 'z1/(z2*sm_0)**2' (:1421), formed once per call from the neutral momentum
    stability function; it converts the divergence into the same units as the deformation.

    THREE FORTRAN LOOPS, FUSED. ':1483' forms the strain velocity in the scratch array 'hlp',
    ':1509' cubes it over the length scale, and ':1521' copies it to 'tket_hshr'. The two
    intermediates never reach a savepoint -- 'hlp' is overwritten later in the same section by
    the SSO wake production -- so nothing is lost by fusing them.

    WHAT THE STEPS ARE. 'hlp' after :1483 is a strain velocity: a length scale times the trace of
    the two-dimensional strain-rate tensor of the separated mode. Raschendorfer's comments call
    that out -- ":1472 not equal to trace of 2D-strain tensor" on the branch this port does not
    take, ":1483 equal to trace of 2D-strain tensor" on the one it does -- and the difference is
    exactly the '-wert' subtraction, which removes the divergent part that incompressibility
    already assigns to the vertical. A velocity cubed over a length is a TKE production rate
    [m2/s3], which is :1509, and it enters the budget as such.

    'imode_shshear' SELECTS THIS FORM as well: at 0, :1472 takes the former variant
    'hor_scale*SQRT(hdef2 + hdiv**2)', which is a different number by orders of magnitude here.
    'test_the_capture_uses_the_trace_constrained_horizontal_strain' establishes which one ran.

    'loutshshr' GATES ONLY THE COPY at :1521, not the computation: with the output switch off the
    production is still formed and still added to 'frm' at :1534, just not published. The port
    writes the output field either way, because the total forcing is its other consumer.
    """
    scaled_divergence = (
        wpfloat("1.0")
        / (
            (wpfloat("2.0") * neutral_momentum_stability_function)
            * (wpfloat("2.0") * neutral_momentum_stability_function)
        )
    ) * horizontal_divergence
    strain_velocity = effective_horizontal_shear_length_scale * (
        sqrt(scaled_divergence * scaled_divergence + horizontal_deformation_square)
        - scaled_divergence
    )
    return (
        strain_velocity * strain_velocity * strain_velocity
    ) / effective_horizontal_shear_length_scale


@gtx.field_operator
def _compute_sso_wake_energy_production(
    sso_tendency_u: fa.CellKField[wpfloat],
    sso_tendency_v: fa.CellKField[wpfloat],
    wind_u: fa.CellKField[wpfloat],
    wind_v: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """The scalar product of the SSO wind tendency with the wind [m2/s3].

    The SSO scheme's wind tendency projected onto the wind itself is the rate at which the
    sub-grid orography drains kinetic energy from the resolved flow, per unit mass. It is
    negative wherever the SSO scheme is doing its job -- the drag opposes the wind -- which is
    why the total forcing negates it and floors it at zero.

    MAIN LEVELS, NOT HALF LEVELS. 'u', 'v', 'ut_sso' and 'vt_sso' are main-level fields and the
    Fortran loop runs 'DO k=1,kem', one row further up than everything else in the section. The
    staggering is resolved by the next statement, which interpolates onto the half levels.

    IT LANDS IN THE SCRATCH ARRAY 'hlp', overwriting the separated-shear source the Fortran left
    there, and it is what 'hlp' holds at the section's exit savepoint. That is safe because the
    shear source has already been copied to 'tket_hshr'; in the port the two are separate fields
    and the ordering constraint disappears with the aliasing that caused it.

    THE BLOCK IS GUARDED by 'ltkemcsso .OR. loutmcsso' (:1553) and by '.NOT. lini' (:1550), the
    latter because 'ut_sso' and 'vt_sso' may not have been computed yet during initialisation.
    In practice it runs exactly when the SSO scheme does.

    THIS EXPRESSION IS AN FMA CANARY. It is 'a*b + c*d', the pattern that distinguishes a
    contracted build from an uncontracted one, so a bit-exact result here is evidence that
    neither side is fusing -- the role section 1b)'s '_compute_thermal_forcing' plays there.
    """
    return sso_tendency_u * wind_u + sso_tendency_v * wind_v


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def add_three_dimensional_shear_complements(
    vertical_gradient_u: fa.CellKField[wpfloat],
    vertical_gradient_v: fa.CellKField[wpfloat],
    dwdx: fa.CellKField[wpfloat],
    dwdy: fa.CellKField[wpfloat],
    horizontal_divergence: fa.CellKField[wpfloat],
    horizontal_deformation_square: fa.CellKField[wpfloat],
    min_forcing: fa.CellField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    horizontal_mesh_size: fa.CellField[wpfloat],
    horizontal_shear_length_factor: wpfloat,
    karman_constant: wpfloat,
    half_level_height: fa.CellKField[wpfloat],
    surface_height: fa.CellField[wpfloat],
    turbulent_velocity_scale: fa.CellKField[wpfloat],
    neutral_momentum_stability_function: wpfloat,
    sso_tendency_u: fa.CellKField[wpfloat],
    sso_tendency_v: fa.CellKField[wpfloat],
    wind_u: fa.CellKField[wpfloat],
    wind_v: fa.CellKField[wpfloat],
    mean_shear_forcing: fa.CellKField[wpfloat],
    inverse_richardson_number_factor: fa.CellKField[wpfloat],
    uncorrected_horizontal_shear_length_scale: fa.CellField[wpfloat],
    effective_horizontal_shear_length_scale: fa.CellKField[wpfloat],
    separated_horizontal_shear_tke_source: fa.CellKField[wpfloat],
    sso_wake_energy_production: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute everything section 2a) forms before it adds to 'frm', in six statements.

    Args:
        vertical_gradient_u: Vertical gradient of the zonal wind at the mass centre,
            'zvari(:,:,u_m)' [1/s], half levels.
        vertical_gradient_v: Vertical gradient of the meridional wind, 'zvari(:,:,v_m)' [1/s].
        dwdx: Zonal derivative of the vertical wind, 'dwdx' [1/s], half levels.
        dwdy: Meridional derivative of the vertical wind, 'dwdy' [1/s], half levels.
        horizontal_divergence: Horizontal wind divergence, 'hdiv' [1/s], half levels.
        horizontal_deformation_square: Squared horizontal deformation, 'hdef2' [1/s2].
        min_forcing: Lower limit of the TKE forcing, 'fc_min' [1/s2], one value per column.
        thermal_forcing: The buoyancy forcing 'frh' [1/s2], written by section 1b) and not
            touched here.
        horizontal_mesh_size: 'l_hori' [m], one value per column.
        horizontal_shear_length_factor: 'a_hshr' [-].
        karman_constant: 'akt' [-].
        half_level_height: 'hhl' [m], half levels.
        surface_height: 'hhl(:,ke1)' [m], one value per column; a fixed absolute level, which a
            relative GT4Py offset cannot express.
        turbulent_velocity_scale: 'tke(:,:,nvor)' [m/s] at the previous time level, half levels.
        neutral_momentum_stability_function: 'sm_0' [-], a derived closure constant.
        sso_tendency_u: 'ut_sso' [m/s2], main levels.
        sso_tendency_v: 'vt_sso' [m/s2], main levels.
        wind_u: 'u' [m/s], main levels at the mass centre.
        wind_v: 'v' [m/s], main levels at the mass centre.
        mean_shear_forcing: Output [1/s2], the Fortran's 'ftm': 'frm' after the two mean-flow
            loops and before the non-turbulent contributions. No savepoint holds it.
        inverse_richardson_number_factor: Output, 'xri' [-], stored with 'ke' rows.
        uncorrected_horizontal_shear_length_scale: Output, 'layr' [m], one value per column.
        effective_horizontal_shear_length_scale: Output, 'hor_scale' [m], stored with 'ke' rows.
        separated_horizontal_shear_tke_source: Output, 'tket_hshr' [m2/s3], half levels.
        sso_wake_energy_production: Output, 'hlp' [m2/s3], MAIN levels.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k=2'.
        vertical_end: End of the half levels; 'ke', mirroring 'k=...,kem' with 'kem = ke'. The
            surface half level 'ke1' is not written by this section.
    """
    _compute_three_dimensional_shear_forcing(
        vertical_gradient_u=vertical_gradient_u,
        vertical_gradient_v=vertical_gradient_v,
        dwdx=dwdx,
        dwdy=dwdy,
        horizontal_divergence=horizontal_divergence,
        horizontal_deformation_square=horizontal_deformation_square,
        min_forcing=min_forcing,
        out=mean_shear_forcing,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_inverse_richardson_number_factor(
        mean_shear_forcing=mean_shear_forcing,
        thermal_forcing=thermal_forcing,
        out=inverse_richardson_number_factor,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_uncorrected_horizontal_shear_length_scale(
        horizontal_mesh_size=horizontal_mesh_size,
        horizontal_shear_length_factor=horizontal_shear_length_factor,
        karman_constant=karman_constant,
        out=uncorrected_horizontal_shear_length_scale,
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )
    _compute_effective_horizontal_shear_length_scale(
        uncorrected_horizontal_shear_length_scale=uncorrected_horizontal_shear_length_scale,
        half_level_height=half_level_height,
        surface_height=surface_height,
        inverse_richardson_number_factor=inverse_richardson_number_factor,
        turbulent_velocity_scale=turbulent_velocity_scale,
        out=effective_horizontal_shear_length_scale,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_separated_horizontal_shear_tke_source(
        effective_horizontal_shear_length_scale=effective_horizontal_shear_length_scale,
        horizontal_divergence=horizontal_divergence,
        horizontal_deformation_square=horizontal_deformation_square,
        neutral_momentum_stability_function=neutral_momentum_stability_function,
        out=separated_horizontal_shear_tke_source,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_sso_wake_energy_production(
        sso_tendency_u=sso_tendency_u,
        sso_tendency_v=sso_tendency_v,
        wind_u=wind_u,
        wind_v=wind_v,
        out=sso_wake_energy_production,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start - 1, vertical_end),
        },
    )
