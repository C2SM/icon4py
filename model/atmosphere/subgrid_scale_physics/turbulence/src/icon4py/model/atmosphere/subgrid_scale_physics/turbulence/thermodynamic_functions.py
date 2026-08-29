# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Moist thermodynamic helpers shared by the turbulence stencils.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90' at icon commit 26d6b98cce, the
commit that produced the reference capture: the four statement functions at the foot of the
module (:3382-3442), SUBROUTINE 'turb_cloud' (:1966-2218) and the thermodynamic tail of
SUBROUTINE 'adjust_satur_equil' (:920-1043). The scientific commentary in that file is by
Matthias Raschendorfer (DWD).

Nothing here is a '@gtx.program'. These are the pieces two callers share -- 'turbdiff' section
0) calls 'adjust_satur_equil' twice, once for the main levels and once for the lower boundary of
the Prandtl layer, and 'turbtran' calls it again -- so they live in one module rather than being
written out in each stencil that needs them. Field operators inline, so sharing them costs
nothing at run time.

BIT-EXACTNESS STOPS HERE. Every quantity below that passes through 'exp' or 'log' disagrees
with the ICON reference in the last bits: nvhpc's libm and the libm behind GT4Py round the
exponential differently, by up to one ULP, and 'turb_cloud' then subtracts two nearly equal
numbers ('dq = qt - qs') and amplifies that. The gates of the stencils that use these functions
are 'Tol' with reason 'TRANSCENDENTAL' for that reason and no other; see the measurements in
'tests/turbulence/integration_tests/test_turbdiff_section_0.py'.
"""

import enum

import gt4py.next as gtx
from gt4py.next import exp, log, maximum, minimum, where

from icon4py.model.common import constants, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


class ThermoConstants(ta.wpfloat, enum.Enum):
    """The physical constants the moist thermodynamics of the scheme reads.

    An enum rather than plain module-level names because GT4Py can only fold a closure variable
    that is an ATTRIBUTE of an enum class or of an 'eve.FrozenNamespace'
    ('foast_passes/closure_var_folding.py'); a bare module-level float reaches the program
    lowering as an unresolved symbol, and a module attribute -- 'constants.RD_O_CPD' -- makes
    'foast_to_gtir.py::visit_Attribute' raise "Unreachable". This is the pattern
    'MicrophysicsConstants' already uses.

    The member names are the Fortran's, so that each expression below lines up with the
    statement it was translated from.
    """

    #: 'grav', the gravitational acceleration [m/s2].
    GRAV = constants.GRAV
    #: 'edgrav = z1/grav' (turb_diffusion.f90:903), the reciprocal gravitational acceleration
    #: [s2/m]. The Fortran multiplies 'gz0' by this rather than dividing by 'grav', which is one
    #: rounding elsewhere, so 'compute_turbulent_length_scale' does the same.
    EDGRAV = 1.0 / constants.GRAV
    #: 'r_d', the gas constant of dry air [J/K/kg].
    RD = constants.RD
    #: 'rdocp = rd_o_cpd = rd/cpd' [-].
    RDOCP = constants.RD_O_CPD
    #: 'p0ref', the reference pressure of the Exner function [Pa].
    P0REF = constants.P0REF
    #: 'rvd_m_o = vtmpc1 = rv/rd - 1' [-].
    RVD_M_O = constants.RV_O_RD_MINUS_1
    #: 'rdv = rd/rv', the ratio of the two gas constants (mo_granules_phys.f90:30) [-].
    RDV = constants.RD / constants.RV
    #: 'lhocp = alvdcp = alv/cpd', the latent heat of vaporisation in temperature units
    #: (mo_physical_constants.f90:145) [K].
    LHOCP = constants.LATENT_HEAT_FOR_VAPORISATION / constants.CPD
    #: 'b3 = tmelt', the melting point of ice [K].
    B3 = constants.MELTING_TEMPERATURE
    #: 'b1 = c1es' [Pa], the saturation vapour pressure at the melting point in Magnus' formula
    #: (mo_granules_phys.f90:108). The three that follow are the DWD coefficients selected by
    #: 'itype_satpres_coeffs = 1' (mo_nwp_phy_nml.f90:256), which is the namelist default and
    #: what exp.mch_icon-ch2_small runs; 'itype_satpres_coeffs = 2' would select the IFS
    #: coefficients and is not ported.
    C1ES = 610.78
    #: 'b2w = c3les', the numerator coefficient over water [-].
    C3LES = 17.269
    #: 'b4w = c4les', the offset in the denominator over water [K].
    C4LES = 35.86
    #: 'b234w = c5les = c3les*(tmelt - c4les)', the combination the derivative needs [K].
    C5LES = 17.269 * (constants.MELTING_TEMPERATURE - 35.86)
    #: 'rsig_max', the maximal RELATIVE standard deviation of the local super-saturation
    #: (turb_utilities.f90:2065). A Fortran PARAMETER, not a namelist entry. Its absolute
    #: companion 'asig_max' belongs to 'imode_stadlim = 1', which is not ported.
    RSIG_MAX = 0.05


@gtx.field_operator
def _exner_factor(pressure: fa.CellKField[wpfloat]) -> fa.CellKField[wpfloat]:
    """The Exner factor '(p/p0)**(R_d/c_pd)' [-]; 'zexner', turb_utilities.f90:3382-3390.

    The Fortran writes the power as 'EXP(rdocp*LOG(zpres/p0ref))' rather than as '**', and so
    does this: '**' would lower to 'math.pow', which is a different function of the same
    mathematical value and rounds differently (see the package README, "Bit-exactness").
    """
    return exp(ThermoConstants.RDOCP * log(pressure / ThermoConstants.P0REF))


@gtx.field_operator
def _saturation_vapour_pressure(temperature: fa.CellKField[wpfloat]) -> fa.CellKField[wpfloat]:
    """Saturation vapour pressure over water [Pa] by Magnus' formula; 'zpsat_w', :3392-3399."""
    return ThermoConstants.C1ES * exp(
        ThermoConstants.C3LES
        * (temperature - ThermoConstants.B3)
        / (temperature - ThermoConstants.C4LES)
    )


@gtx.field_operator
def _saturation_specific_humidity(
    vapour_pressure: fa.CellKField[wpfloat],
    dry_air_pressure: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Saturation specific humidity from a vapour and a dry-air partial pressure [kg/kg].

    'zqvap', turb_utilities.f90:3410-3423 -- the "new version", which takes the partial pressure
    of DRY air where 'zqvap_old' took the total pressure ("mod_2011/09/28: zpres=patm ->
    zpres=pdry"). The Fortran assigns the numerator to the result variable first and divides in
    a second statement; that is the same two roundings as the single expression here.
    """
    vapour = ThermoConstants.RDV * vapour_pressure
    return vapour / (dry_air_pressure + vapour)


@gtx.field_operator
def _dqsat_dt(
    temperature: fa.CellKField[wpfloat],
    saturation_specific_humidity: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Temperature derivative of the saturation specific humidity [1/K]; 'zdqsdt', :3432-3442.

    The square in the denominator is written as a product on purpose: Fortran's '(x)**2' with
    an integer literal is a multiplication, GT4Py's '**' is 'math.pow', and CUDA's 'pow' carries
    up to 2 ULP.
    """
    denominator = temperature - ThermoConstants.C4LES
    return (
        ThermoConstants.C5LES
        * (wpfloat("1.0") - saturation_specific_humidity)
        * saturation_specific_humidity
        / (denominator * denominator)
    )


@gtx.field_operator
def _diagnose_cloud_cover_and_liquid_water(  # noqa: PLR0917  [too-many-positional] -- GT4Py field operators are called positionally
    pressure: fa.CellKField[wpfloat],
    liquid_water_temperature: fa.CellKField[wpfloat],
    total_water: fa.CellKField[wpfloat],
    supersaturation_deviation: fa.CellKField[wpfloat],
    critical_normalized_supersaturation: wpfloat,
    cloud_cover_at_saturation: wpfloat,
    relative_accuracy_limit: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Statistical saturation adjustment: saturation fraction and liquid water content.

    SUBROUTINE 'turb_cloud' (turb_utilities.f90:1966-2218) at 'icldtyp = 2' and
    'imode_stadlim = 2'. Raschendorfer's own description of the method (:2007-2013):

        "A Gaussian distribution is assumed for the local super-saturation 'dq := qt - qs' where
         'qt := qv + ql' is the total water content and 'qs' is the saturation specific humidity.
         Using the standard deviation of this distribution SDSS (as input) and the
         quasi-conservative quantities 'qt' and 'tl' (liquid water temperature), a corrected
         liquid water content is determined, which contains also the contributions by
         subgrid-scale cloud processes. A corresponding cloudiness is calculated as well."

    ONLY ONE OF THE ROUTINE'S FIVE BRANCHES IS TRANSLATED. 'icldtyp' selects between grid-scale
    adjustment (0), an empirical relative-humidity criterion (1) and this statistical adjustment
    (2); 'adjust_satur_equil' derives it from 'icldm_turb' and 'itype_wcld', which are 2 and 2
    in every configuration this port targets ('itype_wcld' is frozen at 2 in 'TurbulenceConfig',
    and 'icldm_turb = 2' selects this call site at all -- at 'icldm_turb <= 1' the caller does
    not reach 'turb_cloud'). 'imode_stadlim' is frozen at 2, the relative limit on the SDSS;
    'imode_stadlim = 1' would cap it absolutely at 'asig_max' instead.

    The three parameters are the namelist quantities the Fortran combines at the top of the
    routine, and they are derived here rather than passed pre-combined so that the arithmetic
    stays the Fortran's:

        zclc0  = MIN(clc_diag, 1 - epsi)      cloud cover at which the parametrization saturates
        zq_max = q_crit*(1/zclc0 - 1)         normalized super-saturation of full cloud cover
        zq_inv = 1/(zq_max + q_crit)          reciprocal width of the transition

    Args:
        pressure: total air pressure [Pa]
        liquid_water_temperature: 'tl', the temperature the parcel would have if all its cloud
            water evaporated [K]
        total_water: 'qt', vapour plus cloud water [kg/kg]
        supersaturation_deviation: 'rcld' on input, the standard deviation of the local
            super-saturation carried over from the previous call of the scheme [kg/kg]
        critical_normalized_supersaturation: 'q_crit' [-]
        cloud_cover_at_saturation: 'clc_diag' [-]
        relative_accuracy_limit: 'epsi' [-]

    Returns:
        saturation fraction (cloud cover) [-], and the liquid water content [kg/kg]
    """
    zero = wpfloat("0.0")
    one = wpfloat("1.0")
    saturating_cover = minimum(cloud_cover_at_saturation, one - relative_accuracy_limit)
    maximal_normalized_supersaturation = critical_normalized_supersaturation * (
        one / saturating_cover - one
    )
    inverse_transition_width = one / (
        maximal_normalized_supersaturation + critical_normalized_supersaturation
    )

    # 'pdry': the partial pressure of dry air. "mod_2011/09/28: zpres=patm -> zpres=pdry".
    dry_air_pressure = (
        (one - total_water) / (one + ThermoConstants.RVD_M_O * total_water) * pressure
    )
    saturation_humidity = _saturation_specific_humidity(
        _saturation_vapour_pressure(liquid_water_temperature), dry_air_pressure
    )
    # 'gam': the slope factor of the saturation adjustment, the fraction of an excess of total
    # water that condenses rather than raising the saturation humidity with the released heat.
    condensation_slope = one / (
        one + ThermoConstants.LHOCP * _dqsat_dt(liquid_water_temperature, saturation_humidity)
    )
    supersaturation = total_water - saturation_humidity

    # 'imode_stadlim = 2': the SDSS is capped RELATIVE to the saturation humidity. 'qs' is then
    # reused for the in-cloud water content of a fully covered box, which is why the Fortran
    # overwrites it; both names below say which is meant.
    deviation = minimum(ThermoConstants.RSIG_MAX * saturation_humidity, supersaturation_deviation)
    saturated_water_content = minimum(total_water, deviation * maximal_normalized_supersaturation)

    # 'q', the super-saturation normalized by its standard deviation. Where the deviation
    # vanishes the distribution is a spike and the diagnosis degenerates to the grid-scale
    # saturation adjustment: fully clear below saturation, fully cloudy above it.
    normalized_supersaturation = where(
        deviation <= zero,
        where(
            supersaturation <= zero,
            -critical_normalized_supersaturation,
            maximal_normalized_supersaturation,
        ),
        supersaturation / deviation,
    )
    cloud_cover = minimum(
        one,
        maximum(
            zero,
            (normalized_supersaturation + critical_normalized_supersaturation)
            * inverse_transition_width,
        ),
    )
    liquid_water = (
        condensation_slope
        * where(
            normalized_supersaturation >= maximal_normalized_supersaturation,
            supersaturation,
            saturated_water_content,
        )
        * (cloud_cover * cloud_cover)
    )
    return cloud_cover, liquid_water


@gtx.field_operator
def _thermodynamic_factors(  # noqa: PLR0917  [too-many-positional] -- GT4Py field operators are called positionally
    liquid_water_temperature: fa.CellKField[wpfloat],
    total_water: fa.CellKField[wpfloat],
    liquid_water: fa.CellKField[wpfloat],
    cloud_cover: fa.CellKField[wpfloat],
    exner_factor: fa.CellKField[wpfloat],
    pressure: fa.CellKField[wpfloat],
    cloud_cover_shape_factor: wpfloat,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """The tail of 'adjust_satur_equil': the factors the buoyancy production is built from.

    turb_utilities.f90:920-1043, the part that runs after the cloud diagnosis under
    "lcaltdv = .TRUE., ladjout = .FALSE., icldmod = 2". Raschendorfer's closing note (:1044-1046):

        "Die thermodynamischen Hilfsgroessen wurden hier unter Beruecksichtigung der
         diagnostizierten Kondensationskorrektur gebildet, indem die entspr. korrigierten Werte
         fuer t, qv und qc benutzt wurden. Beachte, dass es zu einer gewissen Inkonsistenz kommt,
         wenn die Dichte nicht angepasst wird!"

        -- "The thermodynamic auxiliary quantities were formed here taking the diagnosed
           condensation correction into account, by using the correspondingly corrected values
           of t, qv and qc. Note that a certain inconsistency arises if the density is not
           adjusted as well."

    The three intermediate quantities the Fortran keeps in the aliased output arrays are, in its
    own words, the "corrected temperature" 'temp' (the temperature the box actually has, once
    the diagnosed cloud water has released its latent heat), the "corrected water vapor" 'qvap',
    and 'virt', the "rezipr. virtual factor" -- the reciprocal of the virtual-temperature
    correction, so that 'virt*p' is the pressure a dry parcel of the same density would have.

    'mcor' is the "moist correction by turb. phase-transitions": the part of a buoyancy
    fluctuation that comes from condensation inside the turbulent eddy rather than from the
    conserved variables themselves. It is proportional to the cloud cover, so it vanishes in
    clear air and both buoyancy factors reduce to their dry forms.

    'r_cpd = c_p/c_pd' is the constant 1 here: it is 1 unless 'lcpfluc' is set, and
    'TurbulenceConfig' freezes that switch at '.FALSE.' ("fluctuations of the heat capacity of
    air not considered"). It is returned rather than dropped because 'zaux(:,:,2)' is a measured
    output of section 0) and later sections read it.

    Args:
        liquid_water_temperature: 'tet_liq' as the routine has it at this point -- still a
            LIQUID WATER TEMPERATURE [K], not yet divided by the Exner factor
        total_water: 'q_h2o' [kg/kg]
        liquid_water: 'q_liq', the diagnosed liquid water content [kg/kg]
        cloud_cover: the saturation fraction from the cloud diagnosis [-]
        exner_factor: the Exner factor at the same levels [-]
        pressure: total air pressure [Pa]
        cloud_cover_shape_factor: 'c_scld' [-]

    Returns:
        the liquid-water potential temperature [K], 'dQ_sat/dT' [1/K], the buoyancy factor of
        the 'tet_l' gradient [m/s2/K], the buoyancy factor of the 'h2o_g' gradient [m/s2], the
        effective cloud cover [-] and the air density [kg/m3]. The density is only kept by the
        surface caller, which is the one that asks 'adjust_satur_equil' for it
        ('lcalrho = .TRUE.'); the main-level caller receives ICON's own 'rhoh' instead and
        discards this.
    """
    one = wpfloat("1.0")
    corrected_temperature = liquid_water_temperature + ThermoConstants.LHOCP * liquid_water
    corrected_vapour = total_water - liquid_water
    reciprocal_virtual_factor = one / (
        one + ThermoConstants.RVD_M_O * corrected_vapour - liquid_water
    )
    reduced_pressure = reciprocal_virtual_factor * pressure

    # The Fortran turns 'tet_liq' from a temperature into a potential temperature here, between
    # the density and the saturation derivative, and this keeps that position even though
    # nothing between the two reads it.
    liquid_water_potential_temperature = liquid_water_temperature / exner_factor

    dry_air_pressure = (one - corrected_vapour) * reduced_pressure
    dqsat_dt = _dqsat_dt(
        corrected_temperature,
        _saturation_specific_humidity(
            _saturation_vapour_pressure(corrected_temperature), dry_air_pressure
        ),
    )
    effective_cloud_cover = (
        cloud_cover_shape_factor
        * cloud_cover
        / (one + cloud_cover * (cloud_cover_shape_factor - one))
    )
    moist_correction = (
        effective_cloud_cover
        * (
            ThermoConstants.LHOCP / corrected_temperature
            - (one + ThermoConstants.RVD_M_O) * reciprocal_virtual_factor
        )
        / (one + dqsat_dt * ThermoConstants.LHOCP)
    )
    buoyancy_factor_h2o_g = ThermoConstants.GRAV * (
        ThermoConstants.RVD_M_O * reciprocal_virtual_factor + moist_correction
    )
    buoyancy_factor_tet_l = ThermoConstants.GRAV * (
        exner_factor / corrected_temperature - moist_correction * exner_factor * dqsat_dt
    )
    return (
        liquid_water_potential_temperature,
        dqsat_dt,
        buoyancy_factor_tet_l,
        buoyancy_factor_h2o_g,
        effective_cloud_cover,
        reduced_pressure / (ThermoConstants.RD * corrected_temperature),
    )
