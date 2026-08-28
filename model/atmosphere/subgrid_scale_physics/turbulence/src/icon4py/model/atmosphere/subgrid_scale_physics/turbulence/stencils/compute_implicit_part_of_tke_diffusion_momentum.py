"""Implicit part of the diffusion momentum of the vertical TKE diffusion.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'prep_impl_vert_diff' (:2690-2860 at icon commit 26d6b98cce), the branch under
"use precalculated implicit weights" (:2764-2776), which is the branch ICON takes: 'ldynimp'
defaults to '.FALSE.' (mo_turbdiff_config.f90:101) and no configuration under 'icon/run/'
sets it. 'turbdiff' reaches it from section 9), "Aufdatieren des TKE-Profils durch die
(erweiterte) Diffusions-Tendenz" ("Updating the TKE profile by the (extended) diffusion
tendency", turb_diffusion.f90:2407-2489), which calls 'prep_impl_vert_diff' at :2419-2427.
The scientific commentary in those files is by Matthias Raschendorfer (DWD).

The Fortran is one line:

    impl_mom(i,k) = expl_mom(i,k) * tdc%impl_weight(k)

It splits the diffusion momentum 'rho*K/dz' of a flux level into the part the semi-implicit
solve treats implicitly and (in 'subtract_implicit_part_of_tke_diffusion_momentum') the remainder it
treats explicitly. 'impl_weight' is a fixed vertical profile computed once at model
initialisation (mo_nwp_phy_init.f90:1541-1547): 'impl_t' above about 1500 m, ramped linearly
to the over-implicit 'impl_s' at the surface.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_implicit_part_of_tke_diffusion_momentum(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_weight: fa.KField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Weight the diffusion momentum of a flux level by the level's implicit weight."""
    return diffusion_momentum * implicit_weight


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_implicit_part_of_tke_diffusion_momentum(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_weight: fa.KField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'impl_mom', the implicit part of the TKE diffusion momentum, on flux levels.

    The vertical range is the Fortran 'DO k = k_tp+2, k_sf+1-m' with 'k_tp = 1', 'k_sf = ke1'
    and 'm = 1' -- the value of 'm' for the surface-concentration condition the TKE diffusion
    is called with ('lsflucond = .FALSE.', turb_diffusion.f90:2420). So it runs over the whole
    of Fortran 'k = 3..ke1', which is 'vertical_start = 2, vertical_end = nlev + 1' here.

    THIS ROW RANGE IS ONE LONGER THAN THE EXPLICIT PART'S. The surface flux level 'ke1' gets an
    implicit momentum but keeps its full diffusion momentum in 'expl_mom' -- the Fortran says so
    at turb_utilities.f90:2787 ("Notice that 'expl_mom' still contains the whole diffusion
    momentum at level 'k_sf'!") and 'calc_impl_vert_diff' relies on it when it closes the lower
    boundary condition. Writing the two fields in one program would therefore mean rewriting
    'expl_mom(:,ke1)' with its own value, which is why they are two.

    Args:
        diffusion_momentum: 'expl_mom' on entry, the full diffusion momentum 'rho_h * K / dz'
            of the TKE diffusion [kg/m2/s], as section 6) built it. Flux levels, which for the
            TKE diffusion are the main levels: a flux level with a given index sits above the
            half level of the same index.
        implicit_weight: 'tdc%impl_weight' [-], the implicit weight of each flux level; a fixed
            profile from model initialisation, not a field of this scheme.
        implicit_diffusion_momentum: Output, 'impl_mom' [kg/m2/s].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First flux level; 2, mirroring Fortran 'k = 3'.
        vertical_end: End of the flux levels; 'nlev + 1', mirroring Fortran 'k = ..,ke1'.
    """
    _compute_implicit_part_of_tke_diffusion_momentum(
        diffusion_momentum=diffusion_momentum,
        implicit_weight=implicit_weight,
        out=implicit_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
