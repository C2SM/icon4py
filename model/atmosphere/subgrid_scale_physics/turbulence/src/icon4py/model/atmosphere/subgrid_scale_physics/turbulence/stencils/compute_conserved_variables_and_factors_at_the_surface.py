"""The same quantities at the lower boundary of the Prandtl layer, plus Exner factor and density.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section 0),
the second call of 'adjust_satur_equil' (:998-1029 at icon commit 26d6b98cce, the commit that
produced the reference capture), introduced by "Thermodynamische Hilfsvariablen auf Unterrand der
Prandtl-Schicht" -- "thermodynamic auxiliary variables at the lower boundary of the Prandtl
layer". The callee is SUBROUTINE 'adjust_satur_equil', turb_utilities.f90:609-1047. The
scientific commentary in both files is by Matthias Raschendorfer (DWD).

Raschendorfer's note on what the call leaves behind (turb_diffusion.f90:1030-1034):

    "Beachte: 'zvari(:,ke1,tet_l)' und 'zvari(:,ke1,h2o_g)' sind jetzt die Erhaltungsvariablen
     am Unterrand der Prandtl-Schicht (zero-level). Die Werte im Niveau 'ke' wurden dabei zur
     Interpolation benutzt. 'zaux(:,ke1,1)' enthaelt den Exner-Faktor im zero-level."

    -- "Note: 'zvari(:,ke1,tet_l)' and 'zvari(:,ke1,h2o_g)' are now the conserved variables at
       the lower boundary of the Prandtl layer (zero level). The values at level 'ke' were used
       for the interpolation. 'zaux(:,ke1,1)' holds the Exner factor at the zero level."

WHAT THE 'fip' INTERPOLATION IS FOR. The rigid surface is not the level the turbulence scheme
wants a boundary value at; that level is the top of the roughness layer formed by land use, the
"zero level". The conserved variables there are a weighted mean of the surface values and the
lowest atmospheric main level, with the weight 'tfh' -- the fraction of the transfer resistance
that the laminar sub-layer accounts for. Raschendorfer's note on the Exner factor
(turb_utilities.f90:838-842):

    "The roughness layer between the current level (at the rigid surface) and the desired level
     (at the top of the roughness layer) of index 'k=ke1' is assumed to be a pure CFL layer
     without any mass. Consequently the Exner pressure values at both levels are treated like
     being equal!"

which is why one Exner factor, the one built from the surface pressure, appears in both terms of
the temperature interpolation, and why the surface pressure is used unchanged as the pressure of
the zero level.

WHY THE LEVEL ABOVE ARRIVES PRE-SLICED. GT4Py's vertical offsets are relative, and this program's
vertical domain is the single row 'nlev'. Reading the level above with 'Koff[-1]' would work, but
it would mean passing the main-level result as an input while writing the same storage as an
output, which is exactly the Fortran aliasing this port does not reproduce. The caller slices
the two values it needs out of the main-level result instead ('utils.surface_row'), which is the
convention the package already uses for 'tkvm(:,ke1)' and its like.
"""

import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.thermodynamic_functions import (
    ThermoConstants,
    _exner_factor,
    _diagnose_cloud_cover_and_liquid_water,
    _thermodynamic_factors,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_conserved_variables_and_factors_at_the_surface(  # noqa: PLR0917 [too-many-positional-arguments]
    surface_pressure: fa.CellField[wpfloat],
    surface_temperature: fa.CellField[wpfloat],
    surface_specific_humidity: fa.CellField[wpfloat],
    surface_liquid_water: fa.CellField[wpfloat],
    liquid_water_potential_temperature_above: fa.CellField[wpfloat],
    total_water_above: fa.CellField[wpfloat],
    laminar_reduction_factor_for_scalars: fa.CellField[wpfloat],
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
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """Conserved variables, cloud cover, Exner factor, density and factors at the zero level.

    The four surface values are what 'turb_setup' put into the surface row of 'zvari' before
    this section began (turb_utilities.f90:337-361): the surface pressure, the grid-mean surface
    temperature, the surface specific humidity, and a surface liquid water content that is zero
    under 'ilow_def_cond = 2' ("zero surface value as the default lower boundary condition",
    which 'TurbulenceConfig' freezes) and 'qc' of the lowest layer otherwise.

    The flags this call site sets, against the main-level call:

        lcalrho = .TRUE.    the density at the zero level IS computed here; ICON has none
        lcalepr = .TRUE.    so is the Exner factor
        fip     = tfh       PRESENT, so the conserved variables are interpolated (see above)

    Args:
        surface_pressure: 'prss(:,ke1)' = 'ps' [Pa]
        surface_temperature: 'tmps(:,ke1)' = 't_g' [K]
        surface_specific_humidity: 'vaps(:,ke1)' = 'qv_s' [kg/kg]
        surface_liquid_water: 'liqs(:,ke1)' [kg/kg]
        liquid_water_potential_temperature_above: 'zvari(:,ke,tet_l)', the main-level result one
            level up [K]
        total_water_above: 'zvari(:,ke,h2o_g)', likewise [kg/kg]
        laminar_reduction_factor_for_scalars: 'tfh', the interpolation weight 'fip' [-]
        supersaturation_deviation: 'rcld', read at the surface row [kg/kg]
        cloud_cover_shape_factor: 'c_scld' [-]
        critical_normalized_supersaturation: 'q_crit' [-]
        cloud_cover_at_saturation: 'clc_diag' [-]
        relative_accuracy_limit: 'epsi' [-]

    Returns:
        the Exner factor [-], 'tet_l' [K], 'h2o_g' [kg/kg], 'liq' [kg/kg], the cloud cover [-],
        the air density [kg/m3], 'r_cpd' [-], 'dQ_sat/dT' [1/K] and the two buoyancy factors
        [m/s2/K] and [m/s2], all on the single row of the zero level.
    """
    one = wpfloat("1.0")
    pressure = broadcast(surface_pressure, (dims.CellDim, dims.KDim))
    exner_factor = _exner_factor(pressure)

    # The conserved variables at the rigid surface, before the interpolation. 'qc' has no surface
    # value in general, so the Fortran assumes one of zero there and the two definitions below
    # therefore reduce to the surface values themselves whenever 'ilow_def_cond = 2'.
    total_water = surface_specific_humidity + surface_liquid_water
    liquid_water_temperature = surface_temperature - ThermoConstants.LHOCP * surface_liquid_water

    # 'zvari(:,ke,tet_l)' is already a POTENTIAL temperature and 'liquid_water_temperature' is
    # still a plain one, so the Exner factor multiplies the upper term only. That asymmetry is
    # the Fortran's, and it is deliberate (turb_utilities.f90:833-837).
    total_water = broadcast(
        total_water_above * (one - laminar_reduction_factor_for_scalars)
        + total_water * laminar_reduction_factor_for_scalars,
        (dims.CellDim, dims.KDim),
    )
    liquid_water_temperature = (
        exner_factor
        * liquid_water_potential_temperature_above
        * (one - laminar_reduction_factor_for_scalars)
        + liquid_water_temperature * laminar_reduction_factor_for_scalars
    )

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
        air_density,
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
        exner_factor,
        liquid_water_potential_temperature,
        total_water,
        liquid_water,
        effective_cloud_cover,
        air_density,
        broadcast(one, (dims.CellDim, dims.KDim)),
        dqsat_dt,
        buoyancy_factor_tet_l,
        buoyancy_factor_h2o_g,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_conserved_variables_and_factors_at_the_surface(  # noqa: PLR0917 [too-many-positional-arguments]
    surface_pressure: fa.CellField[wpfloat],
    surface_temperature: fa.CellField[wpfloat],
    surface_specific_humidity: fa.CellField[wpfloat],
    surface_liquid_water: fa.CellField[wpfloat],
    liquid_water_potential_temperature_above: fa.CellField[wpfloat],
    total_water_above: fa.CellField[wpfloat],
    laminar_reduction_factor_for_scalars: fa.CellField[wpfloat],
    supersaturation_deviation: fa.CellKField[wpfloat],
    cloud_cover_shape_factor: wpfloat,
    critical_normalized_supersaturation: wpfloat,
    cloud_cover_at_saturation: wpfloat,
    relative_accuracy_limit: wpfloat,
    exner_factor: fa.CellKField[wpfloat],
    liquid_water_potential_temperature: fa.CellKField[wpfloat],
    total_water: fa.CellKField[wpfloat],
    liquid_water: fa.CellKField[wpfloat],
    cloud_cover: fa.CellKField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    specific_heat_ratio: fa.CellKField[wpfloat],
    dqsat_dt: fa.CellKField[wpfloat],
    buoyancy_factor_tet_l: fa.CellKField[wpfloat],
    buoyancy_factor_h2o_g: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_conserved_variables_and_factors_at_the_surface(
        surface_pressure=surface_pressure,
        surface_temperature=surface_temperature,
        surface_specific_humidity=surface_specific_humidity,
        surface_liquid_water=surface_liquid_water,
        liquid_water_potential_temperature_above=liquid_water_potential_temperature_above,
        total_water_above=total_water_above,
        laminar_reduction_factor_for_scalars=laminar_reduction_factor_for_scalars,
        supersaturation_deviation=supersaturation_deviation,
        cloud_cover_shape_factor=cloud_cover_shape_factor,
        critical_normalized_supersaturation=critical_normalized_supersaturation,
        cloud_cover_at_saturation=cloud_cover_at_saturation,
        relative_accuracy_limit=relative_accuracy_limit,
        out=(
            exner_factor,
            liquid_water_potential_temperature,
            total_water,
            liquid_water,
            cloud_cover,
            air_density,
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
