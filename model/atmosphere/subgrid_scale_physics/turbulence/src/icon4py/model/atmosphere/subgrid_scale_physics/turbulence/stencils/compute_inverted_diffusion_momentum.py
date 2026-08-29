# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""LU factorisation of the tridiagonal matrix of the semi-implicit TKE diffusion.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE
'prep_impl_vert_diff', the branch under "without preconditioning" (:2830-2858 at icon commit
26d6b98cce) -- the branch ICON takes, since 'lprecnd' defaults to '.FALSE.'
(mo_turbdiff_config.f90:102) and no configuration under 'icon/run/' sets it. 'turbdiff'
reaches it from section 9) (turb_diffusion.f90:2407-2489, the call at :2419-2427). The
scientific commentary in those files is by Matthias Raschendorfer (DWD).

The Fortran is the top level

    invs_mom(i,k_tp+1) = 1 / (disc_mom(i,k_tp+1) + impl_mom(i,k_tp+2))

followed by the downward recurrence

    invs_fac(i,k) = invs_mom(i,k-1) * impl_mom(i,k)
    invs_mom(i,k) = 1 / (disc_mom(i,k) + impl_mom(i,k+1) + impl_mom(i,k)*(1 - invs_fac(i,k)))

which is the forward elimination of a Thomas solve: the tridiagonal matrix of the diffusion
equation has diagonal 'disc_mom(k) + impl_mom(k) + impl_mom(k+1)' and off-diagonals
'-impl_mom', and 'invs_mom' is the reciprocal of the pivot after elimination. It depends on
the profile being diffused only through the matrix, so it is computed once here and reused by
'solve_tke_diffusion_equation'.

WHY THIS IS A SCAN AND NOT A STENCIL. Unlike most of the sequential 'k'-loops of this scheme
(port spec 3.2), this one really does depend on its own previous output: 'invs_mom(k)' is a
function of 'invs_mom(k-1)'. There is no way to write it as a fixed-depth vertical stencil.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.scan_operator(axis=dims.KDim, forward=True, init=(wpfloat("0.0"), True))
def _invert_diffusion_momentum(
    state: tuple[wpfloat, bool],
    discretisation_momentum: wpfloat,
    implicit_diffusion_momentum: wpfloat,
    implicit_diffusion_momentum_below: wpfloat,
) -> tuple[wpfloat, bool]:
    """One elimination step, carrying the inverted momentum of the level above.

    The carried flag is what distinguishes the first row of the scan, where the Fortran has a
    separate statement without the 'impl_mom(k)*(1 - invs_fac(k))' term: there is no level
    above the uppermost diffused half level, so there is no sub-diagonal to eliminate. Both
    Fortran expressions are written out literally rather than one being folded into the other,
    because 'impl_mom' at that row is not written by this section at all -- it still holds the
    buoyancy factor 'g_tet_l' that shared the storage -- and multiplying it by zero would be a
    numerical accident rather than a boundary condition.

    Returning 'False' unconditionally is what makes the flag true only on the first row.
    """
    inverted_above, at_the_top = state
    inversion_factor = inverted_above * implicit_diffusion_momentum
    inverted = (
        wpfloat("1.0") / (discretisation_momentum + implicit_diffusion_momentum_below)
        if at_the_top
        else wpfloat("1.0")
        / (
            discretisation_momentum
            + implicit_diffusion_momentum_below
            + implicit_diffusion_momentum * (wpfloat("1.0") - inversion_factor)
        )
    )
    return inverted, False


@gtx.field_operator
def _compute_inverted_diffusion_momentum(
    discretisation_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Run the elimination downward, reading the implicit momentum of the flux level below."""
    inverted, _ = _invert_diffusion_momentum(
        discretisation_momentum,
        implicit_diffusion_momentum,
        implicit_diffusion_momentum(Koff[1]),
    )
    return inverted


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_inverted_diffusion_momentum(
    discretisation_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute 'invs_mom', the inverted momentum of the TKE tridiagonal system, on half levels.

    THE STAGGERING. The diffusion routines index a flux level with the same number as the half
    level just below it (turb_diffusion.f90:2413-2417: "In den Diffusionroutinen wird
    vorausgesetzt, dass ein Flussniveau mit gleichem Vertikalindex wie ein Konzentrationsniveau
    gerade ueber letzterem liegt" -- "in the diffusion routines it is assumed that a flux level
    with the same vertical index as a concentration level lies just above the latter"). So the
    two implicit momenta bounding the half level 'k' are 'impl_mom(k)' above and 'impl_mom(k+1)'
    below, which is what the field operator reads with 'Koff[1]'.

    The vertical range is the Fortran 'invs_mom(:,k_tp+1)' statement followed by
    'DO k = k_tp+2, k_sf-m' with 'k_tp = 1', 'k_sf = ke1' and 'm = 1', i.e. Fortran
    'k = 2..ke', which is 'vertical_start = 1, vertical_end = nlev' here. The two Fortran
    blocks are one scan because the first is its initial step, not a different quantity. The
    third Fortran block ('DO k = k_sf+1-m, k_sf-1') is empty at 'm = 1' and is not translated;
    at 'm = 2' it would carry the surface-flux condition, which this scheme does not use for
    the TKE.

    The surface half level 'nlev' is deliberately outside the range: the TKE at the surface is
    a boundary value the transfer scheme sets, not an unknown of this system.

    Args:
        discretisation_momentum: 'disc_mom', 'rho_n * dz / dt_tke' [kg/m2/s] on half levels,
            from section 1a); the mass per unit area of the layer centred on a half level, per
            TKE time step.
        implicit_diffusion_momentum: 'impl_mom' [kg/m2/s] on flux levels, from
            'compute_implicit_part_of_tke_diffusion_momentum'. Rows 'vertical_start + 1' to 'nlev' are
            read; the row at 'vertical_start' is not.
        inverted_diffusion_momentum: Output, 'invs_mom' [m2 s /kg], the reciprocal pivot.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost diffused half level; 1, mirroring Fortran 'k_tp+1 = 2'.
        vertical_end: End of the diffused half levels; 'nlev', mirroring Fortran 'k_sf-m = ke'.
    """
    _compute_inverted_diffusion_momentum(
        discretisation_momentum=discretisation_momentum,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        out=inverted_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
