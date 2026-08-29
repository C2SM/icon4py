# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Quasi-conserved variables, cloud cover and thermodynamic factors on the main levels.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', the first
call of 'adjust_satur_equil' in the section whose banner reads

    "0) Berechnung der Erhaltungsvariablen (auf 'zvari') samt des Bedeckungsgrades und
     thermodynamischer Hilfgroessen, sowie der turbulenten Laengenskalen:"
    -- "0) Calculation of the conserved variables (into 'zvari') together with the cloud cover
        and thermodynamic auxiliary quantities, and of the turbulent length scales:"

(the call is at :969-995 and is introduced by "Thermodynamische Hilfsvariablen auf
Hauptflaechen" -- "thermodynamic auxiliary variables on main levels"; lines are as of icon
commit 26d6b98cce, the commit that produced the reference capture). The callee is
SUBROUTINE 'adjust_satur_equil', turb_utilities.f90:609-1047. The scientific commentary in both
files is by Matthias Raschendorfer (DWD).

WHY THE MAIN LEVELS AND THE SURFACE ARE TWO PROGRAMS. The Fortran calls one subroutine twice
with different flags, and the difference is not a coefficient but a different set of outputs:
the surface call additionally computes the Exner factor and the air density
('lcalepr = lcalrho = .TRUE.'), and it interpolates its conserved variables against the level
above with the Prandtl-layer weight 'tfh' before diagnosing anything. Under the port's
boundary-row rule (spec D14, package README "Boundary rows") a boundary row that writes a
DIFFERENT FIELD stays a program of its own, so these are two. What they do share --
'turb_cloud' and the thermodynamic tail -- is shared as field operators in
'thermodynamic_functions.py'.

Raschendorfer on why there are two calls at all (turb_diffusion.f90:1035-1038):

    "Der 2-te Aufruf von 'adjust_satur_equil' stellt die unteren Randwerte der thermodyn.
     Variablen zur Verfuegung. Dies koennte in den 1-ten Aufruf integriert werden, wenn alle
     thermodyn. Modell-Variablen bis 'k=ke1' allociert waeren. Dies wuerde Vieles vereinfachen!"

    -- "The second call of 'adjust_satur_equil' provides the lower boundary values of the
       thermodynamic variables. This could be merged into the first call if all thermodynamic
       model variables were allocated up to 'k = ke1'. That would simplify a great deal!"

The port keeps them apart for the same reason he could not merge them: ICON's 't', 'qv', 'qc',
'prs' and 'epr' have 'ke' levels and the storage they are written into has 'ke1'.
"""

import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.thermodynamic_functions import (
    ThermoConstants,
    _diagnose_cloud_cover_and_liquid_water,
    _thermodynamic_factors,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_conserved_variables_and_factors_at_main_levels(
    temperature: fa.CellKField[wpfloat],
    specific_humidity: fa.CellKField[wpfloat],
    cloud_water: fa.CellKField[wpfloat],
    pressure: fa.CellKField[wpfloat],
    exner_factor: fa.CellKField[wpfloat],
    supersaturation_deviation: fa.CellKField[wpfloat],
    cloud_cover_shape_factor: wpfloat,
    critical_normalized_supersaturation: wpfloat,
    cloud_cover_at_saturation: wpfloat,
    relative_accuracy_limit: wpfloat,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """Turn the model variables into the set the turbulence closure differentiates.

    The closure needs variables that a phase change does not alter, because a turbulent eddy
    that condenses part of its water must not appear to the scheme as a source of anything.
    Those are the liquid-water temperature 'tl = t - (L/c_p)*q_c' and the total water content
    'qt = q_v + q_c' (turb_utilities.f90:791-798); the liquid-water POTENTIAL temperature
    'tet_l = tl/exner' is what is stored, so that the vertical gradient section 1a) forms is a
    gradient at constant pressure.

    Given those two, the statistical cloud scheme diagnoses the saturation fraction and the
    liquid water content, and the thermodynamic tail turns them into the two buoyancy factors
    that section 1b) multiplies the 'tet_l' and 'h2o_g' gradients by. All of that is in
    'thermodynamic_functions.py'; what is left here is the two definitions above and the
    routing of the flags this call site sets:

        lcalrho = .FALSE.   the density is ICON's own 'rhoh', not recomputed
        lcalepr = .FALSE.   the Exner factor is ICON's own 'epr', not recomputed
        lcaltdv = .TRUE.    the thermodynamic factors ARE wanted
        lpotinp = .FALSE.   the input temperature is a temperature, not a potential one
        ladjout = .FALSE.   the CONSERVED variables are the output, not the adjusted ones
        icldmod = icldm_turb = 2   sub-grid condensation is diagnosed

    'icldm_turb' is the one of the five that a supported configuration can still vary --
    'TurbulenceConfig' documents it as "operationally 1 (DWD global) or 2". ONLY 2 IS
    TRANSLATED. At 'icldm_turb = 1' the Fortran takes a different branch entirely
    (turb_utilities.f90:848-861: the cloud cover is 1 wherever there is cloud water and 0
    elsewhere, and the liquid water is ICON's 'qc' unchanged); at 0 and -1 it takes two more.
    exp.mch_icon-ch2_small sets 2, so those branches have no reference data here and are not
    written rather than being written untested.

    Args:
        temperature: 't', on main levels [K]
        specific_humidity: 'qv', water vapour mixing ratio on main levels [kg/kg]
        cloud_water: 'qc', cloud water content on main levels [kg/kg]
        pressure: 'prs', total air pressure on main levels [Pa]
        exner_factor: 'epr', ICON's own Exner factor on main levels [-]
        supersaturation_deviation: 'rcld' on input, the standard deviation of the local
            super-saturation the previous call of the scheme left behind [kg/kg]
        cloud_cover_shape_factor: 'c_scld' [-]
        critical_normalized_supersaturation: 'q_crit' [-]
        cloud_cover_at_saturation: 'clc_diag' [-]
        relative_accuracy_limit: 'epsi' [-]

    Returns:
        'tet_l' the liquid-water potential temperature [K], 'h2o_g' the total water content
        [kg/kg], 'liq' the liquid water content [kg/kg], the cloud cover [-], 'r_cpd' the
        specific-heat ratio [-], 'dQ_sat/dT' [1/K], and the two buoyancy factors [m/s2/K] and
        [m/s2].
    """
    total_water = specific_humidity + cloud_water
    liquid_water_temperature = temperature - ThermoConstants.LHOCP * cloud_water

    cloud_cover, liquid_water = _diagnose_cloud_cover_and_liquid_water(
        pressure=pressure,
        liquid_water_temperature=liquid_water_temperature,
        total_water=total_water,
        supersaturation_deviation=supersaturation_deviation,
        critical_normalized_supersaturation=critical_normalized_supersaturation,
        cloud_cover_at_saturation=cloud_cover_at_saturation,
        relative_accuracy_limit=relative_accuracy_limit,
    )
    (
        liquid_water_potential_temperature,
        dqsat_dt,
        buoyancy_factor_tet_l,
        buoyancy_factor_h2o_g,
        effective_cloud_cover,
        _air_density,
    ) = _thermodynamic_factors(
        liquid_water_temperature=liquid_water_temperature,
        total_water=total_water,
        liquid_water=liquid_water,
        cloud_cover=cloud_cover,
        exner_factor=exner_factor,
        pressure=pressure,
        cloud_cover_shape_factor=cloud_cover_shape_factor,
    )
    return (
        liquid_water_potential_temperature,
        total_water,
        liquid_water,
        effective_cloud_cover,
        broadcast(wpfloat("1.0"), (dims.CellDim, dims.KDim)),
        dqsat_dt,
        buoyancy_factor_tet_l,
        buoyancy_factor_h2o_g,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_conserved_variables_and_factors_at_main_levels(
    temperature: fa.CellKField[wpfloat],
    specific_humidity: fa.CellKField[wpfloat],
    cloud_water: fa.CellKField[wpfloat],
    pressure: fa.CellKField[wpfloat],
    exner_factor: fa.CellKField[wpfloat],
    supersaturation_deviation: fa.CellKField[wpfloat],
    cloud_cover_shape_factor: wpfloat,
    critical_normalized_supersaturation: wpfloat,
    cloud_cover_at_saturation: wpfloat,
    relative_accuracy_limit: wpfloat,
    liquid_water_potential_temperature: fa.CellKField[wpfloat],
    total_water: fa.CellKField[wpfloat],
    liquid_water: fa.CellKField[wpfloat],
    cloud_cover: fa.CellKField[wpfloat],
    specific_heat_ratio: fa.CellKField[wpfloat],
    dqsat_dt: fa.CellKField[wpfloat],
    buoyancy_factor_tet_l: fa.CellKField[wpfloat],
    buoyancy_factor_h2o_g: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_conserved_variables_and_factors_at_main_levels(
        temperature=temperature,
        specific_humidity=specific_humidity,
        cloud_water=cloud_water,
        pressure=pressure,
        exner_factor=exner_factor,
        supersaturation_deviation=supersaturation_deviation,
        cloud_cover_shape_factor=cloud_cover_shape_factor,
        critical_normalized_supersaturation=critical_normalized_supersaturation,
        cloud_cover_at_saturation=cloud_cover_at_saturation,
        relative_accuracy_limit=relative_accuracy_limit,
        out=(
            liquid_water_potential_temperature,
            total_water,
            liquid_water,
            cloud_cover,
            specific_heat_ratio,
            dqsat_dt,
            buoyancy_factor_tet_l,
            buoyancy_factor_h2o_g,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
