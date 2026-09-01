# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The horizontal wind on the conserved-variable levels, and the layer depths.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section 0),
the block Raschendorfer heads "Berechnung der horizontalen Windgeschwindigkeiten und
Schichtdicken" -- computation of the horizontal wind speeds and the layer thicknesses --
:1042-1064 at icon commit 26d6b98cce, the commit that produced the reference capture. The
scientific commentary in that file is by Matthias Raschendorfer (DWD).

    DO k=1,ke
      zvari(i,k,u_m) = u(i,k)
      zvari(i,k,v_m) = v(i,k)
      dicke(i,k)     = hhl(i,k) - hhl(i,k+1)
    zvari(i,ke1,u_m) = zvari(i,ke,u_m)*(z1-tfm(i))
    zvari(i,ke1,v_m) = zvari(i,ke,v_m)*(z1-tfm(i))

TWO PROGRAMS UNTIL THE STENCIL MERGE, AND ICON'S OWN BLOCK IS ONE. The wind and the layer depth
have nothing to do with each other numerically; they are one unit because the Fortran computes
them in one loop, and that is the structure a reader of the scheme is looking for. Merging them
does not fuse anything: the two statements write different fields over different ranges.

ONE VERTICAL PAIR, TWO RANGES. The pair is the main levels, 'DO k=1,ke' = '(0, nlev)'; the wind
runs one row deeper, '(vertical_start, vertical_end + 1)', because the zero level below the
Prandtl layer is one of its rows.

'nlev' STAYS A RUNTIME ARGUMENT even though it equals 'vertical_end'. That is 'gt4py-01': a
'concat_where' whose split point is static at the same time as the domain bounds FAILS TO COMPILE on
'dace_cpu'. See the docstring of 'Turbulence._program'.

THE 'concat_where' IS WHY THIS UNIT COSTS 'embedded' SOMETHING. gt4py 1.1.10 cannot execute
'concat_where' on the embedded backend, so the wind's test has always xfailed there and the
layer depth's now does too -- four tests, and that is the whole embedded cost of merging
section 0). It is also why the merge stops here: 'adjust_satur_equil' and 'bound_level_interp',
the other two live blocks of section 0), have no 'concat_where' between them and keep their
embedded cross-check, which is what makes section 0)'s three tolerant gates a
three-backend agreement rather than one backend's word.
"""

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_layer_depth(half_level_height: fa.CellKField[wpfloat]) -> fa.CellKField[wpfloat]:
    """The distance between the two half levels that bound a main level [m].

    'dicke(i,k) = hhl(i,k) - hhl(i,k+1)', positive because ICON numbers levels downward.

    Section 0) is the only section in which the 'dicke' storage holds a geometric depth: section
    1a) overwrites it with the discretisation momentum of the TKE diffusion. Here it is used
    twice -- as the increment of the cumulative height that becomes the turbulent length scale,
    and (through that) nowhere else in this section.

    Rows 0 to 'nlev - 1' are the main levels and the only ones written; run the program with
    'vertical_start = 0', 'vertical_end = nlev'. The surface row keeps what the section found
    there, which is untouched memory.

    Args:
        half_level_height: 'hhl', the height of the model half levels [m]; ICON passes
            'p_metrics%z_ifc'

    Returns:
        the depth of the main layers [m]
    """
    return half_level_height - half_level_height(Koff[1])


@gtx.field_operator
def _wind_including_the_zero_level(
    wind: fa.CellKField[wpfloat],
    laminar_reduction_factor_for_momentum: fa.CellField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """One wind component on the main levels, with its Prandtl-layer boundary value below."""
    return concat_where(
        dims.KDim == nlev,
        wind(Koff[-1]) * (wpfloat("1.0") - laminar_reduction_factor_for_momentum),
        wind,
    )


@gtx.field_operator
def _compute_horizontal_wind_including_the_zero_level(
    zonal_wind: fa.CellKField[wpfloat],
    meridional_wind: fa.CellKField[wpfloat],
    laminar_reduction_factor_for_momentum: fa.CellField[wpfloat],
    nlev: gtx.int32,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Both wind components on the levels of the conserved-variable array.

    Run the program with 'vertical_start = 0', 'vertical_end = nlev + 1'; 'nlev' must be that
    same last row, since it is what selects the reduced value.

    Args:
        zonal_wind: 'u' at the mass centre, main levels [m/s]
        meridional_wind: 'v' at the mass centre, main levels [m/s]
        laminar_reduction_factor_for_momentum: 'tfm' [-]
        nlev: 'ke', the row of the zero level

    Returns:
        'zvari(:,:,u_m)' and 'zvari(:,:,v_m)' [m/s]
    """
    return (
        _wind_including_the_zero_level(zonal_wind, laminar_reduction_factor_for_momentum, nlev),
        _wind_including_the_zero_level(
            meridional_wind, laminar_reduction_factor_for_momentum, nlev
        ),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_wind_and_layer_depth(
    half_level_height: fa.CellKField[wpfloat],
    zonal_wind: fa.CellKField[wpfloat],
    meridional_wind: fa.CellKField[wpfloat],
    laminar_reduction_factor_for_momentum: fa.CellField[wpfloat],
    nlev: gtx.int32,
    layer_depth: fa.CellKField[wpfloat],
    zonal_wind_on_conserved_variable_levels: fa.CellKField[wpfloat],
    meridional_wind_on_conserved_variable_levels: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'dicke' on the main levels and 'zvari(:,:,u_m/v_m)' down to the zero level.

    Args:
        half_level_height: 'hhl' [m], half levels.
        zonal_wind: 'u' at the mass centre, main levels [m/s].
        meridional_wind: 'v' at the mass centre, main levels [m/s].
        laminar_reduction_factor_for_momentum: 'tfm' [-], one value per column.
        nlev: 'ke', the row of the zero level; equal to 'vertical_end' and passed separately for
            'gt4py-01'.
        layer_depth: Output, 'dicke' [m], main levels. The surface row is not written here --
            section 1a) overwrites the whole array with 'rho_n*dz/dt' before anything reads it.
        zonal_wind_on_conserved_variable_levels: Output, 'zvari(:,:,u_m)' [m/s].
        meridional_wind_on_conserved_variable_levels: Output, 'zvari(:,:,v_m)' [m/s].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First main level; 0, mirroring Fortran 'k=1'.
        vertical_end: End of the main levels; 'ke'. The wind runs to 'ke1'.
    """
    _compute_layer_depth(
        half_level_height=half_level_height,
        out=layer_depth,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_horizontal_wind_including_the_zero_level(
        zonal_wind=zonal_wind,
        meridional_wind=meridional_wind,
        laminar_reduction_factor_for_momentum=laminar_reduction_factor_for_momentum,
        nlev=nlev,
        out=(
            zonal_wind_on_conserved_variable_levels,
            meridional_wind_on_conserved_variable_levels,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end + 1),
        },
    )
