"""Geometric depth of the model layers.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section 0),
lines 1043-1053 at icon commit 26d6b98cce, the commit that produced the reference capture. The
loop is headed "Berechnung der horizontalen Windgeschwindigkeiten und Schichtdicken" --
"Calculation of the horizontal wind speeds and the layer thicknesses" -- and the statement itself
carries "Berechnung der Schichtdicken und der Dichte auf Nebenflaechen" ("calculation of the
layer thicknesses and of the density at half levels"). The second half of that comment is stale:
the half-level density is interpolated further down by 'bound_level_interp', not here. The
scientific commentary in that file is by Matthias Raschendorfer (DWD).

The Fortran fuses this with the two wind assignments in one k-loop because they share nothing
but the loop; they are separate programs here, one per quantity.
"""

import gt4py.next as gtx

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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_layer_depth(
    half_level_height: fa.CellKField[wpfloat],
    layer_depth: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_layer_depth(
        half_level_height=half_level_height,
        out=layer_depth,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
