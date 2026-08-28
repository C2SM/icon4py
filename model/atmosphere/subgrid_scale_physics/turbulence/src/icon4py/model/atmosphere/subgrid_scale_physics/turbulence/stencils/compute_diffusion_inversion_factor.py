"""Inversion factor of the tridiagonal matrix of the semi-implicit TKE diffusion.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'prep_impl_vert_diff' (:2842 at icon commit 26d6b98cce), reached from 'turbdiff' section 9)
(turb_diffusion.f90:2407-2489, the call at :2419-2427). The scientific commentary in those
files is by Matthias Raschendorfer (DWD).

The Fortran is one line of the elimination loop:

    invs_fac(i,k) = invs_mom(i,k-1) * impl_mom(i,k)

the multiplier of the forward elimination -- the sub-diagonal entry times the reciprocal pivot
of the level above -- which the back substitution of 'solve_tke_diffusion_equation' then needs.

WHY THIS IS A SEPARATE PROGRAM AND NOT AN OUTPUT OF THE SCAN. It is the same product
'compute_inverted_diffusion_momentum' forms inside its scan, and could have been carried out
of it as a second component. It is not, because the two have different vertical ranges: the
inverted momentum is defined one half level higher, where there is no level above and hence no
multiplier, and a scan cannot write its two outputs on two different domains. Recomputing one
multiplication is cheaper than the alternative, which is writing a row the Fortran leaves
alone -- 'frh' at that row still holds section 6)'s flux density.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_diffusion_inversion_factor(
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Multiplier of the forward elimination at one half level."""
    return inverted_diffusion_momentum(Koff[-1]) * implicit_diffusion_momentum


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_diffusion_inversion_factor(
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    inversion_factor: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'invs_fac', the elimination multiplier of the TKE solve, on half levels.

    The vertical range is the Fortran 'DO k = k_tp+2, k_sf-m' with 'k_tp = 1', 'k_sf = ke1' and
    'm = 1', i.e. 'k = 3..ke', which is 'vertical_start = 2, vertical_end = nlev' here. The
    uppermost diffused half level is excluded because there is no level above it to eliminate.

    In 'turbdiff' this lands in the storage of 'frh', which held the flux density of
    circulation kinetic energy until section 8) -- the row above 'vertical_start' still does,
    and must come out unchanged.

    Args:
        inverted_diffusion_momentum: 'invs_mom' [m2 s /kg] on half levels, from
            'compute_inverted_diffusion_momentum'; read one level up.
        implicit_diffusion_momentum: 'impl_mom' [kg/m2/s] on flux levels, from
            'compute_implicit_part_of_tke_diffusion_momentum'.
        inversion_factor: Output, 'invs_fac' [-].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First eliminated half level; 2, mirroring Fortran 'k = 3'.
        vertical_end: End of the diffused half levels; 'nlev', mirroring Fortran 'k_sf-m = ke'.
    """
    _compute_diffusion_inversion_factor(
        inverted_diffusion_momentum=inverted_diffusion_momentum,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        out=inversion_factor,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
