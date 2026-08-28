"""The mass-weighted weight that interpolates a main-level profile onto the half levels.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'bound_level_interp'
(:3232-3367 at icon commit 26d6b98cce, the commit that produced the reference capture), the
'auxil' block at :3291-3302. 'turbdiff' section 0) calls it at turb_diffusion.f90:1084-1085 with
'depth = dp0' and 'auxil = hlp'. The scientific commentary in that file is by Matthias
Raschendorfer (DWD).

    auxil(i,k) = depth(i,k-1) / (depth(i,k-1) + depth(i,k))

The routine has two interpolation branches and this is the one 'auxil' selects: with the weight
precomputed, each of the seven variables costs one multiply-add instead of the four operations of
'zbnd_val'. They are the same number only up to rounding, so which branch was taken is part of
the translation and not an implementation detail -- see 'interpolate_variables_onto_half_levels'.

A comment in the caller (turb_diffusion.f90:1076-1082) records the alternative that was tried and
rejected for the density, which is why 'dp0' and not 'dicke' is the depth:

    "test: different interpolation weights for bl-interpolation of rho:
     Volume-weighted interpolation: ... depth=dicke
     Mass-weighted interpolation (included into multi-variable interpolation:)"

so the weight is a MASS weight: 'dp0' is the pressure thickness of a main layer, and the value at
half level 'k' leans towards the thicker of the two layers that meet there.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_half_level_interpolation_weight(
    layer_pressure_thickness: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """The weight of the LOWER of the two main levels that meet at a half level [-].

    It is the thickness of the UPPER layer over the sum of both, which is what makes the
    interpolated value lean towards the thicker layer: a half level that sits just below a thick
    layer and just above a thin one is nearly the thin layer's value.

    Only rows 1 to 'nlev - 1' are written -- the Fortran loop is 'DO k = ke, 2, -1' -- so run the
    program with 'vertical_start = 1', 'vertical_end = nlev'. There is nothing to write at the
    model top or at the surface: neither half level lies between two main levels.

    Args:
        layer_pressure_thickness: 'dp0', the pressure thickness of the main layers [Pa]

    Returns:
        'hlp' as section 0) leaves it, the interpolation weight [-]
    """
    thickness_above = layer_pressure_thickness(Koff[-1])
    return thickness_above / (thickness_above + layer_pressure_thickness)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_half_level_interpolation_weight(
    layer_pressure_thickness: fa.CellKField[wpfloat],
    interpolation_weight: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_half_level_interpolation_weight(
        layer_pressure_thickness=layer_pressure_thickness,
        out=interpolation_weight,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
