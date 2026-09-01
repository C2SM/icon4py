# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The tridiagonal matrix of one VARIABLE TYPE of 'vertdiff', built and LU-factorised.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90', SUBROUTINE 'vert_grad_diff'
(:2461-2478 at icon commit 26d6b98cce) and SUBROUTINE 'prep_impl_vert_diff' (:2764-2849), which
'vertdiff' reaches at turb_vertdiff.f90:747-763. The scientific commentary in those files is by
Matthias Raschendorfer (DWD).

ONE MATRIX SERVES EVERY VARIABLE OF ITS TYPE, which is the whole reason 'vertdiff' loops over
types on the outside and variables on the inside: 'vert_grad_diff' factorises only under
'linisetup .OR. lnewvtype' (:2401), so the three scalars -- temperature, water vapour, cloud
water -- are solved with one 'invs_mom' and one 'invs_fac', and so are the two wind components.

ONE PROGRAM, SIX STATEMENTS, FOUR DOMAINS. In source order,

  1. the interior diffusion momentum 'expl_mom = tkv*rhon/diff_dep', flux levels 2..ke;
  2. the surface diffusion momentum 'expl_mom(:,k_sf) = rhon*tsv';
  3. the surface diffusion depth 'diff_dep(:,k_sf) = tkv/tsv';
  4. the implicit part 'impl_mom = expl_mom*impl_weight', 2..'k_sf+1-m';
  5. the explicit part, 'expl_mom = expl_mom - impl_mom', IN PLACE, 2..ke;
  6. the forward elimination 'invs_mom', a scan, 1..'k_sf-m'.

THE TWO TYPES DIFFER BY ONE ROW AND BY NOTHING ELSE, and 'elimination_end' is that one row. The
Fortran writes it as 'm', 1 for a surface-CONCENTRATION condition and 2 for a surface-FLUX
condition, and every range that depends on it is 'k_sf + 1 - m' or 'k_sf - m'. Rather than bind
the program twice, 'elimination_end' is an ordinary integer argument -- one past the last half
level the elimination writes, 'nlev' for the momentum type and 'nlev - 1' for the scalar type --
and statements 4 and 6 are expressed on it. One binding, two callers.

WHAT IS DELIBERATELY NOT HERE. The third Fortran loop, 'DO k = k_sf-m+1, k_sf-1', finishes the
row the elimination stopped at under a flux condition and is empty under a concentration one;
and the multiplier 'invs_fac' follows both. Neither is a statement of this program, and the
reason is measured rather than aesthetic: both read 'invs_mom' through 'Koff[-1]' and the first
of them WRITES 'invs_mom'. A statement whose 'out=' names the same program parameter as a
SHIFTED input is silently dropped by DaCe -- see the module docstring of 'solve_turb_budgets' for
the five-variant measurement -- so it cannot be a statement here, and the multiplier has to
follow it rather than precede it. They stay two programs, called in that order, and one of them
('compute_diffusion_inversion_factor') was already shared with 'turbdiff' section 9).

Statement 5 is in place and that IS admissible: it reads its output POINTWISE, which is variant
A3 of the same measurement and is correct on every backend. Its row range is one short of
statement 4's, and the Fortran says why -- "Notice that 'expl_mom' still contains the whole
diffusion momentum at level 'k_sf'!" -- because 'calc_impl_vert_diff' turns that row into the
explicit surface flux, the lower boundary condition of the system.

'_compute_inverted_diffusion_momentum' is IMPORTED rather than copied. It is section 9)'s
operator, and 'prep_impl_vert_diff' really is the same subroutine for the TKE and for the model
variables; it lives in 'prep_impl_vert_diff_for_the_tke', which is section 9)'s flavour of this
very subroutine.
"""

import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.prep_impl_vert_diff_for_the_tke import (
    _compute_inverted_diffusion_momentum,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_diffusion_momentum(
    diffusion_coefficient: fa.CellKField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    diffusion_depth: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'tkv*rhon/diff_dep' [kg/m2/s], the product formed before the quotient as in the Fortran."""
    return diffusion_coefficient * air_density / diffusion_depth


@gtx.field_operator
def _compute_surface_diffusion_momentum(
    air_density: fa.CellKField[wpfloat],
    surface_transfer_velocity: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'rhon(k_sf)*tsv' [kg/m2/s]."""
    return air_density * broadcast(surface_transfer_velocity, (dims.CellDim, dims.KDim))


@gtx.field_operator
def _compute_surface_diffusion_depth(
    diffusion_coefficient: fa.CellKField[wpfloat],
    surface_transfer_velocity: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'tkv(k_sf)/tsv' [m]."""
    return diffusion_coefficient / broadcast(surface_transfer_velocity, (dims.CellDim, dims.KDim))


@gtx.field_operator
def _compute_implicit_diffusion_momentum(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_weight: fa.KField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'expl_mom(k)*impl_weight(k)' [kg/m2/s]."""
    return diffusion_momentum * implicit_weight


@gtx.field_operator
def _subtract_implicit_diffusion_momentum(
    diffusion_momentum: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'expl_mom(k) - impl_mom(k)' [kg/m2/s]."""
    return diffusion_momentum - implicit_diffusion_momentum


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def prep_impl_vert_diff(
    diffusion_coefficient: fa.CellKField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    surface_transfer_velocity: fa.CellField[wpfloat],
    implicit_weight: fa.KField[wpfloat],
    discretisation_momentum: fa.CellKField[wpfloat],
    elimination_end: gtx.int32,
    diffusion_momentum: fa.CellKField[wpfloat],
    diffusion_depth: fa.CellKField[wpfloat],
    implicit_diffusion_momentum: fa.CellKField[wpfloat],
    inverted_diffusion_momentum: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Build and factorise the matrix of one variable type.

    THE FOUR VERTICAL RANGES. '(vertical_start, vertical_end)' is the Fortran
    'DO k = k_hi+1, k_lw' with 'k_hi = 1' and 'k_lw = ke', i.e. '(1, nlev)', and statements 1
    and 5 use it unchanged. The others are

      * the surface flux level '(vertical_end, vertical_end + 1)', Fortran 'k_sf = ke1', for
        statements 2 and 3;
      * '(vertical_start, elimination_end + 1)' for the implicit part, Fortran
        'DO k = k_tp+2, k_sf+1-m';
      * '(vertical_start - 1, elimination_end)' for the elimination, Fortran
        'invs_mom(:,k_tp+1)' followed by 'DO k = k_tp+2, k_sf-m'.

    'diffusion_depth' arrives with its interior rows from
    'prepare_the_vertical_diffusion_matrix', which computes them once for the whole stage;
    statement 1 reads those rows and statement 3 writes the surface row, which is per type. The
    two ranges are disjoint.

    'diffusion_coefficient' is read at the surface row as well as inside, so that row must be
    the one 'turbtran' produced and 'turbdiff' left alone -- section 4) writes rows 1..nlev-1
    only.

    Args:
        diffusion_coefficient: 'vtyp(ivtype)%tkv', i.e. 'tkvm' or 'tkvh' [m2/s].
        air_density: 'rhon' on half levels, surface row included [kg/m3].
        surface_transfer_velocity: 'vtyp(ivtype)%tsv', i.e. 'tvm' or 'tvh' [m/s], one value per
            column.
        implicit_weight: 'tdc%impl_weight(k)' [-], one value per level.
        discretisation_momentum: 'disc_mom' [kg/m2/s], from
            'prepare_the_vertical_diffusion_matrix'.
        elimination_end: One past the last half level the forward elimination writes: 'nlev' for
            a surface-concentration condition ('m = 1', the momentum type) and 'nlev - 1' for a
            surface-flux condition ('m = 2', the scalar type). The implicit part runs one row
            further than this.
        diffusion_momentum: In-out, 'expl_mom' [kg/m2/s]. Written whole by statements 1 and 2,
            then reduced to its explicit part in place by statement 5 -- except on the surface
            flux level, which keeps the whole momentum.
        diffusion_depth: In-out, 'diff_dep' [m]. Interior rows are read; the surface row is
            written.
        implicit_diffusion_momentum: Output, 'impl_mom' [kg/m2/s].
        inverted_diffusion_momentum: Output, 'invs_mom' [m2 s/kg], the reciprocal pivots of the
            elimination -- complete only after
            'invert_diffusion_momentum_at_the_surface_flux_level' has run under a flux
            condition.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: Uppermost interior flux level; 1, mirroring Fortran 'k = 2'.
        vertical_end: End of the interior flux levels; 'nlev', mirroring Fortran 'k_lw = ke'.
    """
    _compute_diffusion_momentum(
        diffusion_coefficient=diffusion_coefficient,
        air_density=air_density,
        diffusion_depth=diffusion_depth,
        out=diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_surface_diffusion_momentum(
        air_density=air_density,
        surface_transfer_velocity=surface_transfer_velocity,
        out=diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end, vertical_end + 1),
        },
    )
    _compute_surface_diffusion_depth(
        diffusion_coefficient=diffusion_coefficient,
        surface_transfer_velocity=surface_transfer_velocity,
        out=diffusion_depth,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end, vertical_end + 1),
        },
    )
    _compute_implicit_diffusion_momentum(
        diffusion_momentum=diffusion_momentum,
        implicit_weight=implicit_weight,
        out=implicit_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, elimination_end + 1),
        },
    )
    _subtract_implicit_diffusion_momentum(
        diffusion_momentum=diffusion_momentum,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        out=diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_inverted_diffusion_momentum(
        discretisation_momentum=discretisation_momentum,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        out=inverted_diffusion_momentum,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start - 1, elimination_end),
        },
    )
