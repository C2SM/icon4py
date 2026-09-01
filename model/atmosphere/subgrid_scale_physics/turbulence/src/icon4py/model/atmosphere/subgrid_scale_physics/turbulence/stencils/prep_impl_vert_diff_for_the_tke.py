# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""'prep_impl_vert_diff' for the TKE: the LU factorisation of section 9)'s tridiagonal matrix.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'prep_impl_vert_diff'
(:2690-2860 at icon commit 26d6b98cce, the commit that produced the reference capture), as
'turbdiff' section 9) calls it (turb_diffusion.f90:2419-2427, inside "Aufdatieren des
TKE-Profils durch die (erweiterte) Diffusions-Tendenz"). The scientific commentary in those
files is by Matthias Raschendorfer (DWD).

ICON USES ONE SUBROUTINE FOR THE TKE AND FOR THE MODEL VARIABLES, and so does the port: this is
the TKE flavour, 'prep_impl_vert_diff' proper is 'vertdiff's. They differ in which branches are
live, not in what they do -- the TKE is called with the surface-CONCENTRATION condition
('lsflucond = .FALSE.', turb_diffusion.f90:2420), so 'm = 1' and there is no
'invert_diffusion_momentum_at_the_surface_flux_level' step here.

FOUR STATEMENTS, in the Fortran's order:

    :2770  impl_mom(i,k) = expl_mom(i,k) * tdc%impl_weight(k)      k = k_tp+2 .. k_sf+1-m
    :2783  expl_mom(i,k) = expl_mom(i,k) - impl_mom(i,k)           k = k_tp+2 .. k_sf-1
    :2834  invs_mom(i,k_tp+1) = 1/(disc_mom + impl_mom(k_tp+2))    and the downward recurrence
    :2842  invs_fac(i,k)      = invs_mom(i,k-1) * impl_mom(i,k)    k = k_tp+2 .. k_sf-1

Each of the four was a '@gtx.program' of its own until the stencil merge. Only the branches ICON
takes are ported: 'ldynimp' is '.FALSE.' (mo_turbdiff_config.f90:101) so the implicit weights
are precalculated, and 'lprecnd' is '.FALSE.' (:102) so there is no preconditioning -- which
'test_section_9_writes_only_the_solver_state' establishes from the data, since 'td_frm' would
otherwise carry 'scal_fac'.

ONE VERTICAL PAIR, THREE RANGES DERIVED FROM IT. The pair the granule binds is the solve's own
'k_tp+1 .. k_sf-1', which is '(1, nlev)'. The implicit split runs one row lower and one row
deeper, '(vertical_start + 1, vertical_end + 1)'; the in-place reduction and the inversion factor
run '(vertical_start + 1, vertical_end)'; the elimination scan runs the bound pair itself.

THE SPLIT AND THE REDUCTION DO NOT COVER THE SAME ROWS, and the Fortran says why on the line
after: "Notice that 'expl_mom' still contains the whole diffusion momentum at level 'k_sf'!"
The surface flux level keeps its full diffusion momentum, and 'calc_impl_vert_diff_for_the_tke'
reads it there to close the lower boundary condition. As two statements with two domains that
is simply true; as one 'concat_where' it would have been a rewrite of a row with its own value.

THE ALIASING RULES THIS PROGRAM MEETS, all three measured in 'solve_turb_budgets', whose
docstring carries the five-variant table:

  * statement 1 READS 'explicit_diffusion_momentum' and statement 2 WRITES it, in place. One
    parameter, not two, so GT4Py sees the dependency and every backend orders it; and the write
    is POINTWISE, which is the only in-place shape DaCe compiles correctly. 'prep_impl_vert_diff'
    does the same thing with the same parameter for the model variables.
  * statements 3 and 4 read 'implicit_diffusion_momentum', which statement 1 wrote, at
    'Koff[+1]' and pointwise; statement 4 reads 'inverted_diffusion_momentum', which statement 3
    wrote, at 'Koff[-1]'. Reading a parameter an EARLIER statement wrote is ordinary dataflow and
    is correct at any offset.
  * nothing here is aliased by the caller.

'_compute_diffusion_inversion_factor' IS IMPORTED, not copied: 'compute_diffusion_inversion_
factor' is still a '@gtx.program' of its own because 'vertdiff' calls it after
'invert_diffusion_momentum_at_the_surface_flux_level', which cannot be a statement. One operator,
two callers, exactly as ICON has one subroutine.

THE SCAN IS DEFINED HERE AND 'vertdiff' IMPORTS IT. '_invert_diffusion_momentum' and
'_compute_inverted_diffusion_momentum' were 'compute_inverted_diffusion_momentum.py' before the
merge, and 'prep_impl_vert_diff' already imported them from there; that module's own
'@gtx.program' had no caller left once this one absorbed it, so the operators moved here with it.

WHY THE ELIMINATION IS A SCAN AND NOT A STENCIL. Unlike most of the sequential 'k'-loops of this
scheme (port spec 3.2), this one really does depend on its own previous output: 'invs_mom(k)' is
a function of 'invs_mom(k-1)'. There is no fixed-depth vertical stencil for it.
"""

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_diffusion_inversion_factor import (
    _compute_diffusion_inversion_factor,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_implicit_part_of_tke_diffusion_momentum(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_weight: fa.KField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Weight the diffusion momentum of a flux level by the level's implicit weight."""
    return diffusion_momentum * implicit_weight


@gtx.field_operator
def _subtract_implicit_part_of_tke_diffusion_momentum(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Take the implicit part out of the diffusion momentum, leaving the explicit remainder."""
    return diffusion_momentum - implicit_diffusion_momentum


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
def prep_impl_vert_diff_for_the_tke(
    implicit_weight: fa.KField[wpfloat],
    discretisation_momentum: fa.CellKField[wpfloat],
    explicit_diffusion_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
    inversion_factor: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Factorise the tridiagonal matrix of the TKE diffusion, in four statements.

    Args:
        implicit_weight: 'tdc%impl_weight' [-], the implicit weight of each flux level; a fixed
            profile from model initialisation (mo_nwp_phy_init.f90:1541-1547), not a field of
            this scheme.
        discretisation_momentum: 'disc_mom', 'rho_n * dz / dt_tke' [kg/m2/s] on half levels,
            from section 1a). Read, never written.
        explicit_diffusion_momentum: 'expl_mom' [kg/m2/s] on flux levels. IN PLACE: it arrives
            holding the full diffusion momentum 'rho_h * K / dz' as section 6) built it, and
            leaves holding the explicit remainder -- except at the surface flux level, which
            keeps the full value.
        implicit_diffusion_momentum: Output, 'impl_mom' [kg/m2/s] on flux levels, the
            sub-diagonal of the system.
        inverted_diffusion_momentum: Output, 'invs_mom' [m2 s/kg], the reciprocal of the pivot
            after elimination.
        inversion_factor: Output, 'invs_fac' [-], the multiplier of the forward elimination.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost diffused half level, Fortran 'k_tp+1' with 'k_tp = 1'; 1.
        vertical_end: End of the diffused half levels, Fortran 'k_sf-1' with 'k_sf = ke1';
            'nlev'.
    """
    _compute_implicit_part_of_tke_diffusion_momentum(
        diffusion_momentum=explicit_diffusion_momentum,
        implicit_weight=implicit_weight,
        out=implicit_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start + 1, vertical_end + 1),
        },
    )
    _subtract_implicit_part_of_tke_diffusion_momentum(
        diffusion_momentum=explicit_diffusion_momentum,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        out=explicit_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start + 1, vertical_end),
        },
    )
    _compute_inverted_diffusion_momentum(
        discretisation_momentum=discretisation_momentum,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        out=inverted_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_diffusion_inversion_factor(
        inverted_diffusion_momentum=inverted_diffusion_momentum,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        out=inversion_factor,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start + 1, vertical_end),
        },
    )
