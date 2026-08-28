"""Explicit part of the diffusion momentum of the vertical TKE diffusion.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'prep_impl_vert_diff' (:2778-2787 at icon commit 26d6b98cce), reached from 'turbdiff' section
9) (turb_diffusion.f90:2407-2489, the call at :2419-2427). The scientific commentary in those
files is by Matthias Raschendorfer (DWD).

The Fortran is one line, in place:

    expl_mom(i,k) = expl_mom(i,k) - impl_mom(i,k)

so that the array that arrived holding the full diffusion momentum 'rho*K/dz' leaves holding
only the part the semi-implicit solve treats explicitly. The Fortran's own note on the next
line is the reason this program stops one level higher than
'compute_implicit_part_of_tke_diffusion_momentum': "Notice that 'expl_mom' still contains the whole
diffusion momentum at level 'k_sf'!"
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _subtract_implicit_part_of_tke_diffusion_momentum(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Take the implicit part out of the diffusion momentum, leaving the explicit remainder."""
    return diffusion_momentum - implicit_diffusion_momentum


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def subtract_implicit_part_of_tke_diffusion_momentum(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'expl_mom', the explicit part of the TKE diffusion momentum, on flux levels.

    The Fortran works in place and so may the caller: 'diffusion_momentum' and
    'explicit_diffusion_momentum' may be the same field, since every row is a function of its
    own row alone.

    The vertical range is the Fortran 'DO k = k_tp+2, k_sf-1' with 'k_tp = 1' and 'k_sf = ke1',
    i.e. 'k = 3..ke', which is 'vertical_start = 2, vertical_end = nlev' here -- one level
    short of the implicit part. The surface flux level 'ke1' keeps the full diffusion momentum,
    and 'compute_explicit_tke_flux_density' reads it there.

    Args:
        diffusion_momentum: 'expl_mom' on entry, the full diffusion momentum [kg/m2/s].
        implicit_diffusion_momentum: 'impl_mom' [kg/m2/s], from
            'compute_implicit_part_of_tke_diffusion_momentum'.
        explicit_diffusion_momentum: Output, 'expl_mom' on exit [kg/m2/s]; may alias
            'diffusion_momentum'.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First flux level; 2, mirroring Fortran 'k = 3'.
        vertical_end: End of the flux levels; 'nlev', mirroring Fortran 'k = ..,ke'.
    """
    _subtract_implicit_part_of_tke_diffusion_momentum(
        diffusion_momentum=diffusion_momentum,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        out=explicit_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
