# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _gradient_of_conserved_variable(
    variable: fa.CellKField[wpfloat],
    inverse_depth: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Difference one quasi-conserved variable across the layer centred on a half level.

    One expression covers the whole column. The Fortran writes it twice because the reciprocal
    depth it divides by lives in two different arrays:

        zvari(i,ke1,n) = (zvari(i,ke,n)   - zvari(i,ke1,n)) * lays(i,ivtp(n))   ! at the surface
        zvari(i,k,n)   = (zvari(i,k-1,n)  - zvari(i,k,n))   * hlp(i,k)          ! above it

    Here the caller selects the reciprocal depth per row and this operator stays single-valued.
    """
    return (variable(Koff[-1]) - variable) * inverse_depth


@gtx.field_operator
def _compute_vertical_gradients_of_conserved_variables(
    zonal_wind: fa.CellKField[wpfloat],
    meridional_wind: fa.CellKField[wpfloat],
    liquid_water_potential_temperature: fa.CellKField[wpfloat],
    total_water: fa.CellKField[wpfloat],
    liquid_water: fa.CellKField[wpfloat],
    inverse_layer_depth: fa.CellKField[wpfloat],
    surface_transfer_ratio_for_momentum: fa.CellField[wpfloat],
    surface_transfer_ratio_for_scalars: fa.CellField[wpfloat],
    nlev: gtx.int32,
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
    ("Calculation of the required vertical gradients, and storing them in 'zvari'"), lines
    1151-1203 at icon commit 26d6b98cce. Raschendorfer splits it into two blocks, "Am unteren
    Modellrand" ("At the lower boundary of the model", :1162-1170) and "An den darueberliegenden
    Nebenflaechen" ("At the half levels above it", :1186-1203). Both blocks evaluate the same
    difference quotient into the same storage; only the reciprocal depth differs, so they are
    one program here and the row selection is a 'concat_where' on the depth.

    The five variables are the ones turbulence treats as dynamically active, in the Fortran's
    own order (mo_turbdiff_config.f90:62-77): the two horizontal wind components at the mass
    centre, then the three scalars conserved under condensation and evaporation -- liquid-water
    potential temperature, total water and liquid water. Section 0) put them on main levels;
    this section replaces them by their gradients on half levels, from which section 1b) builds
    the thermal and mechanical forcing of the TKE equation.

    THE TWO RECIPROCAL DEPTHS. Above the surface the difference spans a resolved layer, so it is
    divided by that layer's geometric depth and 'inverse_layer_depth' ('hlp', from
    'compute_inverse_layer_depth_and_tke_discretisation_momentum') is its reciprocal. The surface
    row of each variable instead holds its Prandtl-layer lower boundary value, which section 0)
    obtained from the second call of 'adjust_satur_equil' (the winds from the transfer-layer
    reduction 'u(ke)*(1-tfm)'). That difference does not span a resolved layer; it is divided by
    the effective depth of the Prandtl layer, whose reciprocal is the surface transfer ratio from
    'compute_surface_transfer_ratios'. Momentum and scalars have different Prandtl-layer
    resistances, hence two ratios; the Fortran selects between them with 'ivtp(n)', which maps
    the two wind components to 'mom' and 'tet_l', 'h2o_g' and 'liq' to 'sca'.

    THE ROW-WISE OVERWRITE IS NOT A RECURRENCE. The Fortran writes each gradient back into the
    storage its variable came in, sweeping upward within each variable, so 'zvari(k-1)' is still
    the variable when 'zvari(k)' becomes the gradient; likewise the surface block runs before the
    interior one so that 'zvari(ke)' is still the variable when 'zvari(ke1)' is written. Every
    output row is therefore a function of input rows only, which is what makes this an ordinary
    'Koff[-1]' stencil rather than a scan (port spec 3.2). The sweep direction and the block order
    are what keep the aliasing safe in Fortran and mean nothing here, where inputs and outputs are
    separate fields.

    Row 0, the model top, is not written: there is no main level above the first one to difference
    against, which is why the Fortran loop stops at 'k = 2'. Run this with 'vertical_start = 1'
    and 'vertical_end = nlev + 1'; 'nlev' must be that same last row, since it is what selects the
    surface depth.

    Args:
        zonal_wind: 'u_m', zonal wind at the mass centre [m/s], on main levels, with its
            Prandtl-layer boundary value in row 'nlev'
        meridional_wind: 'v_m', meridional wind at the mass centre [m/s], likewise
        liquid_water_potential_temperature: 'tet_l' [K], likewise
        total_water: 'h2o_g', specific total water content [kg/kg], likewise
        liquid_water: 'liq', specific liquid water content [kg/kg], likewise
        inverse_layer_depth: 'hlp', reciprocal depth of the half-level layer [1/m]; read only on
            rows 1 to 'nlev' - 1, which are the rows on which section 1a) defines it
        surface_transfer_ratio_for_momentum: 'lays(:,mom)', reciprocal effective Prandtl-layer
            depth for momentum [1/m]
        surface_transfer_ratio_for_scalars: 'lays(:,sca)', the same for scalars [1/m]
        nlev: 'ke', the index of the surface half level; the one row taking the Prandtl-layer
            depth instead of the geometric one

    Returns:
        the five vertical gradients on half levels: [1/s], [1/s], [K/m], [kg/kg/m], [kg/kg/m]
    """
    inverse_depth_for_momentum = concat_where(
        dims.KDim == nlev, surface_transfer_ratio_for_momentum, inverse_layer_depth
    )
    inverse_depth_for_scalars = concat_where(
        dims.KDim == nlev, surface_transfer_ratio_for_scalars, inverse_layer_depth
    )
    return (
        _gradient_of_conserved_variable(zonal_wind, inverse_depth_for_momentum),
        _gradient_of_conserved_variable(meridional_wind, inverse_depth_for_momentum),
        _gradient_of_conserved_variable(
            liquid_water_potential_temperature, inverse_depth_for_scalars
        ),
        _gradient_of_conserved_variable(total_water, inverse_depth_for_scalars),
        _gradient_of_conserved_variable(liquid_water, inverse_depth_for_scalars),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vertical_gradients_of_conserved_variables(
    zonal_wind: fa.CellKField[wpfloat],
    meridional_wind: fa.CellKField[wpfloat],
    liquid_water_potential_temperature: fa.CellKField[wpfloat],
    total_water: fa.CellKField[wpfloat],
    liquid_water: fa.CellKField[wpfloat],
    inverse_layer_depth: fa.CellKField[wpfloat],
    surface_transfer_ratio_for_momentum: fa.CellField[wpfloat],
    surface_transfer_ratio_for_scalars: fa.CellField[wpfloat],
    nlev: gtx.int32,
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
        surface_transfer_ratio_for_momentum=surface_transfer_ratio_for_momentum,
        surface_transfer_ratio_for_scalars=surface_transfer_ratio_for_scalars,
        nlev=nlev,
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
