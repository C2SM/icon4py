"""Datatests for section 9) of 'turbdiff': the semi-implicit vertical diffusion of the TKE.

The oracle is a real ICON run: 'turbdiff-8-exit' supplies the inputs, 'turbdiff-9-exit' the
expected outputs, for the four timesteps that 'exp.mch_icon-ch2_small' serializes.

The section is two calls -- 'prep_impl_vert_diff' factorises the tridiagonal matrix of the
diffusion equation, 'calc_impl_vert_diff' builds its right-hand side and solves it -- plus one
in-line block that undoes the virtual-profile trick section 8) used to smuggle the circulation
term into the same solve. Seven programs, in the order the caller must run them:

    1  compute_implicit_part_of_tke_diffusion_momentum   'impl_mom'   zaux(:,:,4)
    2  subtract_implicit_part_of_tke_diffusion_momentum  'expl_mom'   zaux(:,:,3)
    3  compute_inverted_diffusion_momentum               'invs_mom'   zaux(:,:,5)
    4  compute_diffusion_inversion_factor                'invs_fac'   frh
    5  compute_explicit_tke_flux_density                 -- (see below)
    6  compute_tke_diffusion_right_hand_side             'eff_flux'   len_scale
    7  solve_tke_diffusion_equation                      'upd_prof'   zaux(:,:,1)
    8  add_virtual_diffusion_increment_to_tke_profile    'upd_prof'   zaux(:,:,1)

WHAT THIS SECTION WRITES
------------------------
'td_zaux', 'td_frh' and 'td_len_scale', and nothing else. That is measured, not read off the
source: 'test_section_9_writes_only_the_solver_state' compares every serialized field between
the two savepoints. In particular 'td_frm' does NOT change, which settles the ambiguity the
savepoint reader flags -- 'prep_impl_vert_diff' would overwrite it with the preconditioning
factor 'scal_fac', but only under 'lprecondi', and 'lprecnd' is off here. Neither does
'td_dicke': the discretisation momentum is read, never written.

'test_section_9_writes_exactly_these_rows' pins the vertical extent of each of them, which is
where this section's translation can go wrong most quietly -- five of the six ranges differ
from each other by one level, and each difference is load-bearing.

WHAT THE 'len_scale' STORAGE HOLDS AFTERWARDS -- NOT AN EFFECTIVE FLUX
----------------------------------------------------------------------
Section 9) points 'eff_flux' at the 'len_scale' array (turb_diffusion.f90:2431) and the
turbulent master length scale is gone from there on. That much is real. What replaces it is
NOT what the Fortran's closing comment (:2448-2450) says, and not what the savepoint reader's
'eff_tke_flux()' docstring says either: the vertical integration that would turn the storage
into "the effective flux densities (positive downward) of the (semi-)implicit vertical
diffusion" runs only under 'leff_flux', which section 9) passes as 'kcm <= ke' (:2434), and
ICON-NWP leaves 'kcm' at 'ke+1'. Measured on this capture: 'kcm = 81', 'ke = 80'.

So the storage holds the right-hand side of the tridiagonal system on the diffused half
levels, and the explicit surface flux on the surface row -- the two intermediate roles the
declaration at turb_utilities.f90:2901-2913 lists before the effective flux. Every field name
in this module and in the stencils follows that, and
'test_the_len_scale_storage_is_not_the_effective_tke_flux' is what keeps the claim measured.

The practical consequence is agreeable: the right-hand side of the solve survives into the
savepoint, so 'solve_tke_diffusion_equation' can be handed ICON's own right-hand side rather
than the one this port computes.

WHY SOME TESTS HERE XFAIL ON 'embedded'
---------------------------------------
Programs 5 and 6 select a boundary row with 'concat_where', which gt4py 1.1.10 cannot run on
the embedded backend (package README, "Boundary rows"), so their tests carry
'uses_concat_where'. The other five programs do not use it, and they keep their embedded
cross-check because every program here takes its inputs from a savepoint rather than from the
program before it -- the convention section 1a) established, and the reason a failure in this
module names one translation instead of a chain.
"""

from __future__ import annotations

from typing import NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.add_virtual_diffusion_increment_to_tke_profile import (
    add_virtual_diffusion_increment_to_tke_profile,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_diffusion_inversion_factor import (
    compute_diffusion_inversion_factor,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_explicit_tke_flux_density import (
    compute_explicit_tke_flux_density,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_implicit_part_of_tke_diffusion_momentum import (
    compute_implicit_part_of_tke_diffusion_momentum,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_inverted_diffusion_momentum import (
    compute_inverted_diffusion_momentum,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_tke_diffusion_right_hand_side import (
    compute_tke_diffusion_right_hand_side,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.solve_tke_diffusion_equation import (
    solve_tke_diffusion_equation,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.subtract_implicit_part_of_tke_diffusion_momentum import (
    subtract_implicit_part_of_tke_diffusion_momentum,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.turbulence import TurbulenceConfig
from icon4py.model.common import dimension as dims
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: The offset provider every stencil in this section needs: all of them difference across
#: neighbouring vertical levels.
_KOFF = {dims.Koff.value: dims.KDim}

#: Zero-based row of the uppermost half level the TKE diffusion solves for, Fortran 'k_tp+1'
#: with 'k_tp = 1' (turb_diffusion.f90:2422). Half level 1 (Fortran) is the model top, where
#: the TKE is not a prognostic unknown; the flux level above the row below it carries no flux,
#: which is the upper boundary condition of the system.
UPPERMOST_DIFFUSED_LEVEL = 1


class Section9(NamedTuple):
    """One timestep of section 9): the savepoints, the bounds and every computed output."""

    entry: sb.IconTurbdiffEntrySavepoint
    before: sb.IconTurbdiffSectionSavepoint
    after: sb.IconTurbdiffSectionSavepoint
    #: 'ke', which as a zero-based row index is the surface half level.
    nlev: int
    #: Half-open range of columns 'turbdiff' actually computed; every comparison is masked with
    #: it, since the rest of the slab is untouched memory holding plausible values.
    columns: slice
    implicit_diffusion_momentum: gtx.Field
    explicit_diffusion_momentum: gtx.Field
    inverted_diffusion_momentum: gtx.Field
    inversion_factor: gtx.Field
    updated_tke_profile: gtx.Field


def _implicit_weight(vct_a: np.ndarray, nlev: int, backend) -> gtx.Field:
    """'tdc%impl_weight', the fixed implicit weight of each flux level.

    It is not a field of the turbulence scheme and it is not serialized: ICON computes it once,
    at model initialisation, in mo_nwp_phy_init.f90:1541-1547, and never changes it. The
    granule will receive it the same way, so the test has to reproduce that initialisation.
    This is that code, one to one:

        DO jk = 1, k1500m          impl_weight(jk) = impl_t
        DO jk = k1500m+1, nlev     impl_weight(jk) = impl_t
                                     + (impl_s - impl_t)*(jk - k1500m)/REAL(nlev - k1500m)
        impl_weight(nlevp1) = impl_s

    "using an over implicit value (impl_s) near surface, reduced to in general slightly
    off-centered value (impl_t) in about 1500 m height", as the comment above it puts it.
    'k1500m' is the index of the first half level at or above 1500 m
    (mo_nwp_phy_init.f90:781-795), taken here from the same reference vertical coordinate
    'vct_a' that ICON took it from rather than hard-coded; for this grid it is 60.

    'impl_s' and 'impl_t' come from the granule's own config defaults, which are
    mo_turbdiff_config.f90's, so that a change to either is a change to this test as well.
    """
    config = TurbulenceConfig()
    k1500m = 1
    for level in range(nlev, 0, -1):  # Fortran 'DO jk = nlev,1,-1', one-based
        if vct_a[level - 1] >= 1500.0 and vct_a[level] < 1500.0:
            k1500m = level
    weight = np.empty(nlev + 1, dtype=float)
    weight[:k1500m] = config.impl_t
    for level in range(k1500m + 1, nlev + 1):
        weight[level - 1] = config.impl_t + (config.impl_s - config.impl_t) * (
            level - k1500m
        ) / float(nlev - k1500m)
    weight[nlev] = config.impl_s
    return gtx.as_field((dims.KDim,), weight, allocator=backend)


def _run_the_matrix_and_the_solve(
    data_provider, grid_savepoint, date: str, backend
) -> Section9:
    """Run the five programs of section 9) that do not select a boundary row.

    Every program is given ICON's own inputs, from the savepoint that holds them, rather than
    the output of the program before it: a defect in one translation then cannot travel into
    another test, and each failure names one program. Where a quantity only exists between the
    two savepoints -- the tridiagonal matrix, the right-hand side -- the savepoint that holds
    it is 'turbdiff-9-exit', because this section is what produces it.

    'compute_explicit_tke_flux_density' and 'compute_tke_diffusion_right_hand_side' are not run
    here: they need 'concat_where', which the embedded backend cannot execute, and keeping them
    out of this runner is what lets the other five keep their embedded cross-check.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="8", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="9", date=date)
    nlev = entry.ke()
    horizontal_start, horizontal_end = gtx.int32(before.ivstart()), gtx.int32(before.ivend())

    implicit_weight = _implicit_weight(grid_savepoint.vct_a().asnumpy(), nlev, backend)

    # 'zaux(:,:,4)' arrives holding the buoyancy factor 'g_tet_l' and leaves holding 'impl_mom'.
    implicit_diffusion_momentum = utils.copy_of(before.g_tet_l(), backend)
    compute_implicit_part_of_tke_diffusion_momentum.with_backend(backend)(
        diffusion_momentum=before.expl_mom(),
        implicit_weight=implicit_weight,
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(2),
        vertical_end=gtx.int32(nlev + 1),
        offset_provider={},
    )

    # 'expl_mom' is updated in place, one flux level short of the implicit part.
    explicit_diffusion_momentum = utils.copy_of(before.expl_mom(), backend)
    subtract_implicit_part_of_tke_diffusion_momentum.with_backend(backend)(
        diffusion_momentum=before.expl_mom(),
        implicit_diffusion_momentum=after.impl_mom(),
        explicit_diffusion_momentum=explicit_diffusion_momentum,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(2),
        vertical_end=gtx.int32(nlev),
        offset_provider={},
    )

    # 'zaux(:,:,5)' arrives holding the buoyancy factor 'g_h2o' and leaves holding 'invs_mom'.
    inverted_diffusion_momentum = utils.copy_of(before.g_h2o(), backend)
    compute_inverted_diffusion_momentum.with_backend(backend)(
        discretisation_momentum=before.disc_mom(),
        implicit_diffusion_momentum=after.impl_mom(),
        inverted_diffusion_momentum=inverted_diffusion_momentum,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(UPPERMOST_DIFFUSED_LEVEL),
        vertical_end=gtx.int32(nlev),
        offset_provider=_KOFF,
    )

    # 'frh' arrives holding section 6)'s CKE flux density and leaves holding 'invs_fac'.
    inversion_factor = utils.copy_of(before.cke_flux_density(), backend)
    compute_diffusion_inversion_factor.with_backend(backend)(
        inverted_diffusion_momentum=after.invs_mom(),
        implicit_diffusion_momentum=after.impl_mom(),
        inversion_factor=inversion_factor,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(2),
        vertical_end=gtx.int32(nlev),
        offset_provider=_KOFF,
    )

    # 'zaux(:,:,1)' arrives holding the Exner factor and leaves holding 'upd_prof'. The solve
    # writes it and the circulation correction rewrites the same rows in place, as the Fortran
    # does; every row there is a function of its own row alone.
    updated_tke_profile = utils.copy_of(before.exner_factor(), backend)
    solve_tke_diffusion_equation.with_backend(backend)(
        # 'eff_tke_flux()' is the savepoint reader's name for this storage; what it actually
        # holds on these rows is the right-hand side of the system being solved, see the
        # module docstring.
        right_hand_side=after.eff_tke_flux(),
        implicit_diffusion_momentum=after.impl_mom(),
        inverted_diffusion_momentum=after.invs_mom(),
        inversion_factor=after.invs_fac(),
        updated_tke_profile=updated_tke_profile,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(UPPERMOST_DIFFUSED_LEVEL),
        vertical_end=gtx.int32(nlev),
        offset_provider=_KOFF,
    )
    add_virtual_diffusion_increment_to_tke_profile.with_backend(backend)(
        saved_tke_profile=before.sav_prof(),
        updated_virtual_profile=updated_tke_profile,
        current_virtual_profile=before.hlp(),
        updated_tke_profile=updated_tke_profile,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(UPPERMOST_DIFFUSED_LEVEL),
        vertical_end=gtx.int32(nlev),
        offset_provider={},
    )

    return Section9(
        entry=entry,
        before=before,
        after=after,
        nlev=nlev,
        columns=slice(before.ivstart(), before.ivend()),
        implicit_diffusion_momentum=implicit_diffusion_momentum,
        explicit_diffusion_momentum=explicit_diffusion_momentum,
        inverted_diffusion_momentum=inverted_diffusion_momentum,
        inversion_factor=inversion_factor,
        updated_tke_profile=updated_tke_profile,
    )


def _run_the_flux_and_the_right_hand_side(
    data_provider, date: str, backend
) -> tuple[gtx.Field, gtx.Field, sb.IconTurbdiffSectionSavepoint, int, slice]:
    """Run the two programs of section 9) that select a boundary row with 'concat_where'.

    The explicit flux density is the one quantity of this section that no savepoint holds in
    full: the Fortran computes it in the storage it is about to overwrite with the right-hand
    side, and only the surface row survives. It is therefore chained into the right-hand side
    here -- there is nothing else to feed it from -- and its output is allocated as NaN rather
    than as an entry state, since it has none.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="8", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="9", date=date)
    nlev = entry.ke()
    horizontal_start, horizontal_end = gtx.int32(before.ivstart()), gtx.int32(before.ivend())

    explicit_tke_flux_density = utils.nan_like(before.mixing_length(), backend)
    compute_explicit_tke_flux_density.with_backend(backend)(
        explicit_diffusion_momentum=after.expl_mom(),
        implicit_diffusion_momentum=after.impl_mom(),
        current_tke_profile=before.hlp(),
        nlev=gtx.int32(nlev),
        explicit_tke_flux_density=explicit_tke_flux_density,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(2),
        vertical_end=gtx.int32(nlev + 1),
        offset_provider=_KOFF,
    )

    # The 'len_scale' storage arrives holding the turbulent master length scale.
    right_hand_side = utils.copy_of(before.mixing_length(), backend)
    compute_tke_diffusion_right_hand_side.with_backend(backend)(
        discretisation_momentum=before.disc_mom(),
        current_tke_profile=before.hlp(),
        explicit_tke_flux_density=explicit_tke_flux_density,
        uppermost_diffused_level=gtx.int32(UPPERMOST_DIFFUSED_LEVEL),
        nlev=gtx.int32(nlev),
        right_hand_side=right_hand_side,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(UPPERMOST_DIFFUSED_LEVEL),
        vertical_end=gtx.int32(nlev + 1),
        offset_provider=_KOFF,
    )
    return (
        explicit_tke_flux_density,
        right_hand_side,
        after,
        nlev,
        slice(before.ivstart(), before.ivend()),
    )


# ------------------------------------------------------------------- what the section writes --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_9_writes_only_the_solver_state(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The section's true output set, measured rather than read off the source.

    'td_frm' is the interesting absence. The savepoint reader refuses to name that storage at
    sections 9) and 10) because 'prep_impl_vert_diff' would overwrite it with the
    preconditioning factor 'scal_fac' -- but only under 'lprecondi', which is not serialized.
    This measurement settles it for this capture: the storage does not change, so
    preconditioning was off and the port is right not to implement it.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="8", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="9", date=date)

    assert utils.fields_that_changed(data_provider, before, after) == {
        "td_zaux",
        "td_frh",
        "td_len_scale",
    }


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_9_writes_exactly_these_rows(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Six outputs, five different vertical ranges, all within one level of each other.

    Getting one of them wrong by a level is the failure mode of this section, and it is not
    visible in a comparison that only looks at the rows the port chose to write. So the ranges
    are measured here directly from the two savepoints, and every stencil test below then
    compares the WHOLE slab against a buffer that started as the entry state, which is what
    turns a wrong range into a failed assertion.

    'zaux(:,:,2)', the saved TKE profile, appears with an empty range: it is read by the
    circulation correction and never written.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="8", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="9", date=date)
    nlev = entry.ke()
    columns = slice(before.ivstart(), before.ivend())

    def rows_that_differ(entry_field, exit_field) -> tuple[int, ...]:
        differs = entry_field[columns] != exit_field[columns]
        return tuple(np.flatnonzero(differs.any(axis=0)).tolist())

    entry_zaux, exit_zaux = before.raw_zaux, after.raw_zaux
    measured = {
        "zaux(:,:,1) upd_prof": rows_that_differ(
            entry_zaux(0).asnumpy(), exit_zaux(0).asnumpy()
        ),
        "zaux(:,:,2) sav_prof": rows_that_differ(
            entry_zaux(1).asnumpy(), exit_zaux(1).asnumpy()
        ),
        "zaux(:,:,3) expl_mom": rows_that_differ(
            entry_zaux(2).asnumpy(), exit_zaux(2).asnumpy()
        ),
        "zaux(:,:,4) impl_mom": rows_that_differ(
            entry_zaux(3).asnumpy(), exit_zaux(3).asnumpy()
        ),
        "zaux(:,:,5) invs_mom": rows_that_differ(
            entry_zaux(4).asnumpy(), exit_zaux(4).asnumpy()
        ),
        "frh invs_fac": rows_that_differ(
            before.cke_flux_density().asnumpy(), after.invs_fac().asnumpy()
        ),
        "len_scale right-hand side": rows_that_differ(
            before.mixing_length().asnumpy(), after.eff_tke_flux().asnumpy()
        ),
    }
    expected = {
        "zaux(:,:,1) upd_prof": tuple(range(UPPERMOST_DIFFUSED_LEVEL, nlev)),
        "zaux(:,:,2) sav_prof": (),
        "zaux(:,:,3) expl_mom": tuple(range(2, nlev)),
        "zaux(:,:,4) impl_mom": tuple(range(2, nlev + 1)),
        "zaux(:,:,5) invs_mom": tuple(range(UPPERMOST_DIFFUSED_LEVEL, nlev)),
        "frh invs_fac": tuple(range(2, nlev)),
        "len_scale right-hand side": tuple(range(UPPERMOST_DIFFUSED_LEVEL, nlev + 1)),
    }
    assert measured == expected


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_len_scale_storage_is_not_the_effective_tke_flux(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The aliased storage holds the solver's right-hand side, not an effective flux density.

    Two independent ways of seeing it, because the Fortran comment at
    turb_diffusion.f90:2448-2450 and the savepoint reader's 'eff_tke_flux()' docstring both
    claim otherwise, and a reader who believes them will misinterpret every value in the array:

      * The block that would produce the effective flux runs under 'leff_flux = (kcm <= ke)'
        (turb_diffusion.f90:2434), and 'kcm' -- the upper bound of the resolved roughness layer
        -- is 'ke1' here, the same 'kcm = ke+1' that makes section 2b) dead. So it did not run.
      * Had it run, its first statement would be the upper zero-flux condition
        'eff_flux(:,k_tp+1) = 0' (turb_utilities.f90:3070-3075). That row is not zero.

    If this test ever fails, a capture with a resolved canopy has appeared, the storage really
    does hold the effective flux, and the port needs the vertical integration this section
    currently does not implement.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="8", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="9", date=date)
    columns = slice(before.ivstart(), before.ivend())

    assert entry.kcm() > entry.ke()
    uppermost = after.eff_tke_flux().asnumpy()[columns, UPPERMOST_DIFFUSED_LEVEL]
    assert (uppermost != 0.0).all()


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_implicit_weight_profile_is_the_one_icon_initialised(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    backend,
) -> None:
    """The one input of this section that no savepoint carries is recovered from the capture.

    'tdc%impl_weight' is a fixed vertical profile from model initialisation, so the section
    test has to construct it. That construction is a place where a wrong constant would be
    invisible -- it would just shift the implicit weighting slightly -- so it is checked
    against the data instead of trusted: 'impl_mom / expl_mom' is the weight, level by level.

    Two properties, both of which a wrong 'k1500m', 'impl_s' or 'impl_t' breaks: the recovered
    ratio is the same in every column (it is a function of the level alone), and it is the
    profile 'mo_nwp_phy_init.f90' builds. The ratio is a quotient of two rounded doubles, so
    the comparison is to within a few ulp; the bit-exact statement is the stencil test below,
    which multiplies rather than divides.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="8", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="9", date=date)
    nlev = entry.ke()
    columns = slice(before.ivstart(), before.ivend())
    levels = slice(2, nlev + 1)

    recovered = (
        after.impl_mom().asnumpy()[columns, levels] / before.expl_mom().asnumpy()[columns, levels]
    )
    spread = recovered.max(axis=0) - recovered.min(axis=0)
    assert (spread <= 8 * np.spacing(recovered.max(axis=0))).all()

    constructed = _implicit_weight(grid_savepoint.vct_a().asnumpy(), nlev, backend).asnumpy()
    np.testing.assert_allclose(
        recovered, np.broadcast_to(constructed[levels], recovered.shape), rtol=1e-14, atol=0.0
    )


# ------------------------------------------------------------------------- the matrix ---------


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_implicit_part_of_tke_diffusion_momentum_agrees_with_icon(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    backend,
) -> None:
    """'impl_mom' over the whole slab, including the two rows the section must not touch.

    The row above the range still holds the buoyancy factor 'g_tet_l' that shared the storage,
    and the comparison is against the exit savepoint, where it still does. This is also what
    pins the implicit weight bit-exactly: it is the only place the profile is multiplied by
    anything.
    """
    run = _run_the_matrix_and_the_solve(data_provider, grid_savepoint, date, backend)

    utils.assert_agrees_with_icon(
        "compute_implicit_part_of_tke_diffusion_momentum",
        "zaux(:,:,4) [impl_mom]",
        run.implicit_diffusion_momentum,
        run.after.impl_mom(),
        columns=run.columns,
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_subtract_implicit_part_of_tke_diffusion_momentum_agrees_with_icon(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    backend,
) -> None:
    """'expl_mom' over the whole slab; the surface flux level must keep the full momentum.

    That row is the reason this is a program of its own rather than a second output of the
    implicit part, and it is the row the whole-slab comparison is here to check: the surface
    flux density reads the full diffusion momentum out of it two programs later.
    """
    run = _run_the_matrix_and_the_solve(data_provider, grid_savepoint, date, backend)

    utils.assert_agrees_with_icon(
        "subtract_implicit_part_of_tke_diffusion_momentum",
        "zaux(:,:,3) [expl_mom]",
        run.explicit_diffusion_momentum,
        run.after.expl_mom(),
        columns=run.columns,
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_inverted_diffusion_momentum_agrees_with_icon(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    backend,
) -> None:
    """The forward elimination of the Thomas solve, bit for bit down the whole column.

    A scan is where a translation is most likely to lose bit-exactness -- the carry can be
    accumulated in a different order than the Fortran's sequential loop -- so this one is worth
    reading as a measurement and not only as a pass. It also checks the first row, whose
    Fortran statement is a separate one outside the loop and is reproduced here by a flag in
    the scan's carry rather than by a boundary expression, so that the embedded backend keeps
    this test.
    """
    run = _run_the_matrix_and_the_solve(data_provider, grid_savepoint, date, backend)

    utils.assert_agrees_with_icon(
        "compute_inverted_diffusion_momentum",
        "zaux(:,:,5) [invs_mom]",
        run.inverted_diffusion_momentum,
        run.after.invs_mom(),
        columns=run.columns,
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_diffusion_inversion_factor_agrees_with_icon(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    backend,
) -> None:
    """'invs_fac' over the whole slab, including the row that is still a CKE flux density.

    The elimination multiplier is defined one half level lower than the inverted momentum it is
    built from, and the row between them is the one this comparison would catch being written.
    """
    run = _run_the_matrix_and_the_solve(data_provider, grid_savepoint, date, backend)

    utils.assert_agrees_with_icon(
        "compute_diffusion_inversion_factor",
        "frh [invs_fac]",
        run.inversion_factor,
        run.after.invs_fac(),
        columns=run.columns,
    )


# ----------------------------------------------------------- the right-hand side (concat_where) --


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_explicit_tke_flux_density_agrees_with_icon_at_the_surface(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The surface row of the explicit flux is the only row of it ICON ever serializes.

    The Fortran computes the whole profile into the storage it is about to overwrite with the
    right-hand side, so the rows above the surface are gone by the time the savepoint is
    written. The surface row is not overwritten, and it is the row that carries the boundary
    condition this program selects with 'concat_where' -- so the one row that can be compared
    is also the one worth comparing. The rows above are validated indirectly, through the
    right-hand side that consumes them.
    """
    flux, _, after, nlev, columns = _run_the_flux_and_the_right_hand_side(
        data_provider, date, backend
    )

    utils.assert_agrees_with_icon(
        "compute_explicit_tke_flux_density",
        "len_scale(:,ke1) [explicit surface flux]",
        flux.asnumpy()[columns, nlev],
        after.eff_tke_flux().asnumpy()[columns, nlev],
    )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_tke_diffusion_right_hand_side_agrees_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The whole 'len_scale' slab as section 9) leaves it: both boundaries and the interior.

    Three rows differ in kind and the comparison covers all of them. The model top still holds
    the turbulent master length scale and must, since the section never writes it. The
    uppermost diffused row drops the flux-convergence term, because there is no flux level
    above it. The surface row is not a right-hand side at all but the explicit flux the
    previous program left there, which this program has to carry through rather than
    recompute.
    """
    _, right_hand_side, after, _, columns = _run_the_flux_and_the_right_hand_side(
        data_provider, date, backend
    )

    utils.assert_agrees_with_icon(
        "compute_tke_diffusion_right_hand_side",
        "len_scale [right-hand side of the TKE diffusion]",
        right_hand_side,
        after.eff_tke_flux(),
        columns=columns,
    )


# ----------------------------------------------------------------------------- the solution ----


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_updated_tke_profile_agrees_with_icon(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    backend,
) -> None:
    """The section's actual product: the TKE profile updated by the diffusion tendency.

    TWO PROGRAMS SHARE ONE ORACLE. 'solve_tke_diffusion_equation' produces the diffused
    VIRTUAL profile and 'add_virtual_diffusion_increment_to_tke_profile' turns it into the true
    one; ICON overwrites the intermediate in place and serializes only the composition, so
    there is nothing to compare the solve against on its own. The assertion is therefore made
    twice, once under each stencil's gate, so that neither program can be downgraded without
    the other being reconsidered.

    The comparison covers the whole slab. The model top and the surface row still hold the
    Exner factor that shared this storage until section 8), and a solve that ran one row too
    far would overwrite one of them.
    """
    run = _run_the_matrix_and_the_solve(data_provider, grid_savepoint, date, backend)

    for stencil in (
        "solve_tke_diffusion_equation",
        "add_virtual_diffusion_increment_to_tke_profile",
    ):
        utils.assert_agrees_with_icon(
            stencil,
            "zaux(:,:,1) [upd_prof]",
            run.updated_tke_profile,
            run.after.upd_prof(),
            columns=run.columns,
        )
