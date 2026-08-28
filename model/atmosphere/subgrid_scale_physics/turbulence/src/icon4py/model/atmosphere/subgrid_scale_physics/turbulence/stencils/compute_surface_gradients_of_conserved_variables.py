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
def _surface_gradient_of_conserved_variable(
    variable: fa.CellKField[wpfloat],
    surface_transfer_ratio: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Turn the Prandtl-layer difference of one variable into its gradient at the surface.

    The Fortran statement, run for each of the 'nmvar' variables in turn:

        zvari(i,ke1,n) = (zvari(i,ke,n) - zvari(i,ke1,n)) * lays(i,ivtp(n))

    The same difference as in the interior -- the value one main level higher minus the value
    here -- scaled by a reciprocal depth. Only the depth differs: geometric in the interior,
    and the effective Prandtl-layer depth here, which is what 'surface_transfer_ratio' is the
    reciprocal of.
    """
    return (variable(Koff[-1]) - variable) * surface_transfer_ratio


@gtx.field_operator
def _compute_surface_gradients_of_conserved_variables(
    zonal_wind: fa.CellKField[wpfloat],
    meridional_wind: fa.CellKField[wpfloat],
    liquid_water_potential_temperature: fa.CellKField[wpfloat],
    total_water: fa.CellKField[wpfloat],
    liquid_water: fa.CellKField[wpfloat],
    surface_transfer_ratio_for_momentum: fa.CellField[wpfloat],
    surface_transfer_ratio_for_scalars: fa.CellField[wpfloat],
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """
    Compute the lower-boundary gradients of the five quasi-conserved model variables.

    Translated from ICON's turb_diffusion.f90, SUBROUTINE 'turbdiff', section
    "1a) Berechnung der benoetigten vertikalen Gradienten und Abspeichern auf 'zvari'"
    ("Calculation of the required vertical gradients"), lines 1162-1170 at icon commit
    26d6b98cce -- the part Matthias Raschendorfer heads "Am unteren Modellrand" ("At the lower
    boundary of the model").

    The surface row of each variable holds its Prandtl-layer lower boundary value, which
    section 0) obtained from the second call of 'adjust_satur_equil' (the winds from the
    transfer-layer reduction 'u(ke)*(1-tfm)'). The difference between the lowest main level and
    that boundary value is not a difference across a resolved layer, so it is not divided by a
    geometric depth: it is divided by the effective depth of the Prandtl layer, whose
    reciprocal is the surface transfer ratio computed in 'compute_surface_transfer_ratios'.

    Momentum and scalars have different Prandtl-layer resistances, hence two ratios; the
    Fortran selects between them with 'ivtp(n)', which maps the two wind components to 'mom'
    and 'tet_l', 'h2o_g' and 'liq' to 'sca'.

    This program writes the single row 'nlev' and nothing else; run it with
    'vertical_start = nlev', 'vertical_end = nlev + 1'. It is independent of
    'compute_vertical_gradients_of_conserved_variables' -- the two write disjoint rows and
    both read only the variables, never each other's output -- so the Fortran's order (this
    one first) is not a dependency.

    Args:
        zonal_wind: 'u_m' [m/s], with its Prandtl-layer boundary value in row 'nlev'
        meridional_wind: 'v_m' [m/s], likewise
        liquid_water_potential_temperature: 'tet_l' [K], likewise
        total_water: 'h2o_g' [kg/kg], likewise
        liquid_water: 'liq' [kg/kg], likewise
        surface_transfer_ratio_for_momentum: reciprocal effective Prandtl-layer depth for
            momentum [1/m]
        surface_transfer_ratio_for_scalars: the same for scalars [1/m]

    Returns:
        the five gradients at the lower boundary: [1/s], [1/s], [K/m], [kg/kg/m], [kg/kg/m]
    """
    return (
        _surface_gradient_of_conserved_variable(zonal_wind, surface_transfer_ratio_for_momentum),
        _surface_gradient_of_conserved_variable(
            meridional_wind, surface_transfer_ratio_for_momentum
        ),
        _surface_gradient_of_conserved_variable(
            liquid_water_potential_temperature, surface_transfer_ratio_for_scalars
        ),
        _surface_gradient_of_conserved_variable(total_water, surface_transfer_ratio_for_scalars),
        _surface_gradient_of_conserved_variable(liquid_water, surface_transfer_ratio_for_scalars),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_gradients_of_conserved_variables(
    zonal_wind: fa.CellKField[wpfloat],
    meridional_wind: fa.CellKField[wpfloat],
    liquid_water_potential_temperature: fa.CellKField[wpfloat],
    total_water: fa.CellKField[wpfloat],
    liquid_water: fa.CellKField[wpfloat],
    surface_transfer_ratio_for_momentum: fa.CellField[wpfloat],
    surface_transfer_ratio_for_scalars: fa.CellField[wpfloat],
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
    _compute_surface_gradients_of_conserved_variables(
        zonal_wind=zonal_wind,
        meridional_wind=meridional_wind,
        liquid_water_potential_temperature=liquid_water_potential_temperature,
        total_water=total_water,
        liquid_water=liquid_water,
        surface_transfer_ratio_for_momentum=surface_transfer_ratio_for_momentum,
        surface_transfer_ratio_for_scalars=surface_transfer_ratio_for_scalars,
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
