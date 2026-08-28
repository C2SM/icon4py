# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _vertical_gradient_of_conserved_variable(
    variable: fa.CellKField[wpfloat],
    inverse_layer_depth: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Difference one quasi-conserved variable across the half-level layer.

    The Fortran statement, run for each of the 'nmvar' variables in turn:

        zvari(i,k,n) = (zvari(i,k-1,n) - zvari(i,k,n)) * hlp(i,k)
    """
    return (variable(Koff[-1]) - variable) * inverse_layer_depth


@gtx.field_operator
def _compute_vertical_gradients_of_conserved_variables(
    zonal_wind: fa.CellKField[wpfloat],
    meridional_wind: fa.CellKField[wpfloat],
    liquid_water_potential_temperature: fa.CellKField[wpfloat],
    total_water: fa.CellKField[wpfloat],
    liquid_water: fa.CellKField[wpfloat],
    inverse_layer_depth: fa.CellKField[wpfloat],
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """
    Compute the vertical gradients of the five quasi-conserved model variables.

    Translated from ICON's turb_diffusion.f90, SUBROUTINE 'turbdiff', section
    "1a) Berechnung der benoetigten vertikalen Gradienten und Abspeichern auf 'zvari'"
    ("Calculation of the required vertical gradients"), lines 1186-1203 at icon commit
    26d6b98cce -- the part Matthias Raschendorfer heads "An den darueberliegenden
    Nebenflaechen" ("At the half levels above it", above the lower boundary treated in
    'compute_surface_gradients_of_conserved_variables').

    The five variables are the ones turbulence treats as dynamically active, in the Fortran's
    own order (mo_turbdiff_config.f90:62-77): the two horizontal wind components at the mass
    centre, then the three scalars conserved under condensation and evaporation -- liquid-water
    potential temperature, total water and liquid water. Section 0) put them on main levels;
    this section replaces them by their gradients on half levels, from which section 1b) builds
    the thermal and mechanical forcing of the TKE equation.

    THE ROW-WISE OVERWRITE IS NOT A RECURRENCE. The Fortran writes each gradient back into the
    storage its variable came in, sweeping upward within each variable, so 'zvari(k-1)' is
    still the variable when 'zvari(k)' becomes the gradient. Every output row is therefore a
    function of input rows only, which is what makes this an ordinary 'Koff[-1]' stencil rather
    than a scan (port spec 3.2). The sweep direction is what keeps the aliasing safe in Fortran
    and means nothing here.

    Row 0, the model top, is not written: there is no main level above the first one to
    difference against, which is why the Fortran loop stops at 'k = 2'. The surface row 'nlev'
    is not written here either; it is a boundary condition rather than a difference quotient
    and has its own program. Run this one with 'vertical_start = 1', 'vertical_end = nlev'.

    Args:
        zonal_wind: 'u_m', zonal wind at the mass centre [m/s], on main levels
        meridional_wind: 'v_m', meridional wind at the mass centre [m/s], on main levels
        liquid_water_potential_temperature: 'tet_l' [K], on main levels
        total_water: 'h2o_g', specific total water content [kg/kg], on main levels
        liquid_water: 'liq', specific liquid water content [kg/kg], on main levels
        inverse_layer_depth: reciprocal depth of the half-level layer [1/m]

    Returns:
        the five vertical gradients on half levels: [1/s], [1/s], [K/m], [kg/kg/m], [kg/kg/m]
    """
    return (
        _vertical_gradient_of_conserved_variable(zonal_wind, inverse_layer_depth),
        _vertical_gradient_of_conserved_variable(meridional_wind, inverse_layer_depth),
        _vertical_gradient_of_conserved_variable(
            liquid_water_potential_temperature, inverse_layer_depth
        ),
        _vertical_gradient_of_conserved_variable(total_water, inverse_layer_depth),
        _vertical_gradient_of_conserved_variable(liquid_water, inverse_layer_depth),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vertical_gradients_of_conserved_variables(
    zonal_wind: fa.CellKField[wpfloat],
    meridional_wind: fa.CellKField[wpfloat],
    liquid_water_potential_temperature: fa.CellKField[wpfloat],
    total_water: fa.CellKField[wpfloat],
    liquid_water: fa.CellKField[wpfloat],
    inverse_layer_depth: fa.CellKField[wpfloat],
    zonal_wind_gradient: fa.CellKField[wpfloat],
    meridional_wind_gradient: fa.CellKField[wpfloat],
    liquid_water_potential_temperature_gradient: fa.CellKField[wpfloat],
    total_water_gradient: fa.CellKField[wpfloat],
    liquid_water_gradient: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_vertical_gradients_of_conserved_variables(
        zonal_wind=zonal_wind,
        meridional_wind=meridional_wind,
        liquid_water_potential_temperature=liquid_water_potential_temperature,
        total_water=total_water,
        liquid_water=liquid_water,
        inverse_layer_depth=inverse_layer_depth,
        out=(
            zonal_wind_gradient,
            meridional_wind_gradient,
            liquid_water_potential_temperature_gradient,
            total_water_gradient,
            liquid_water_gradient,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
