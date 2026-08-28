# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 10) of 'turbdiff', where the q tendency of the TKE diffusion is stored.

'turbdiff-9-exit' supplies the inputs and 'turbdiff-10-exit' the expected outputs, for the four
timesteps 'exp.mch_icon-ch2_small' serializes. The section is 65 lines of Fortran
(turb_diffusion.f90:2490-2554) and one program here:

    1  compute_turbulent_velocity_scale_tendency   'tketens'

THE ENTRY SAVEPOINT IS CONDITIONAL. 'turbdiff-9-exit' could not be lifted out of
'IF (ldotkedif .OR. lcircterm)' -- section 10)'s own banner is inside it -- so it exists only in
a capture where the TKE diffusion or the circulation term runs. It does here ('c_diff = 0.2',
hence 'ldotkedif'), and every test below therefore has data. Nothing in this module may assume
that of a future capture: a run with 'c_diff = 0' and no circulation term takes the ELSE branch
at :2541, resets 'tketens' to zero over rows 1..nlev, and writes no 'turbdiff-9-exit' at all.
That branch is not ported, for want of anything to validate it against. See the Fortran note at
turb_diffusion.f90:2482-2486.

WHAT THIS SECTION WRITES
------------------------
'td_tketens', and nothing else -- measured by 'test_section_10_writes_only_the_q_tendency'.
Rows 1..nlev-1 of it, which is Fortran 'DO k = 2, ke'; the model top is left holding the
advection tendency the routine was called with.

THE SURFACE ROW IS ALREADY ZERO ON ENTRY, so the copy-of-entry convention cannot see the
Fortran's 'tketens(:,ke1) = z0' at all: the row agrees whether the port writes it or not. That
is measured rather than assumed ('test_section_10_writes_exactly_these_rows'), and the gap it
leaves is closed by 'test_the_surface_tendency_is_written_rather_than_left_alone', which
poisons that row of the output buffer with NaN and requires the program to replace it. Without
that test the whole surface boundary of this section would be untested.

WHY THE STENCIL TESTS XFAIL ON 'embedded'
-----------------------------------------
The surface row is a different expression for the same output field, so the package README's
boundary-row rule merges it into one program with 'concat_where', which gt4py 1.1.10 cannot run
on the embedded backend. Section 10) has exactly one program, so all three tests that run it
carry 'uses_concat_where' and the embedded coverage of this section is the three tests that only
read the archive. That is the cost the README's rule names, paid in full here; the compensation
is that the vertical boundary is stated once, inside the stencil, instead of in every caller.
"""

from __future__ import annotations

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_turbulent_velocity_scale_tendency import (
    compute_turbulent_velocity_scale_tendency,
)
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: Zero-based row of the uppermost half level that gets a diffusion tendency, Fortran 'k = 2'.
#: Half level 1 (Fortran) is the model top, which the TKE diffusion never solved for, so there
#: is no updated profile to difference against there.
UPPERMOST_TENDENCY_LEVEL = 1


def _run_the_tendency(data_provider, date: str, backend, *, poison: str | None = None):
    """Run section 10)'s only program on one timestep.

    Args:
        data_provider: The serialized archive.
        date: One of 'utils.TURBDIFF_DATES'.
        backend: The backend under test.
        poison: Which NaN poisoning to apply, for the two tests that measure what the program
            does NOT touch. 'output-surface' fills the surface row of the output buffer, which
            the program must overwrite with zero; 'input-surface' fills the surface row of
            'upd_prof', which the surface branch must not read. Either one leaves a NaN in the
            result unless the claim holds, and the caller compares the whole slab bit-exactly.

    Returns:
        The computed 'tketens', the exit savepoint, and the computed-column window.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="9", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="10", date=date)
    nlev = entry.ke()

    updated_tke_profile = before.upd_prof()
    if poison == "input-surface":
        values = updated_tke_profile.asnumpy().copy()
        values[:, nlev] = np.nan
        updated_tke_profile = gtx.as_field(updated_tke_profile.domain, values, allocator=backend)

    tendency = utils.copy_of(before.tketens(), backend)
    if poison == "output-surface":
        values = before.tketens().asnumpy().copy()
        values[:, nlev] = np.nan
        tendency = gtx.as_field(before.tketens().domain, values, allocator=backend)

    compute_turbulent_velocity_scale_tendency.with_backend(backend)(
        updated_tke_profile=updated_tke_profile,
        turbulent_velocity_scale=before.tke(),
        inverse_tke_time_step=entry.fr_tke(),
        nlev=gtx.int32(nlev),
        turbulent_velocity_scale_tendency=tendency,
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        vertical_start=gtx.int32(UPPERMOST_TENDENCY_LEVEL),
        vertical_end=gtx.int32(nlev + 1),
        offset_provider={},
    )
    return tendency, after, slice(before.ivstart(), before.ivend())


# ------------------------------------------------------------------- what the section writes --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_10_writes_only_the_q_tendency(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """One output slot, measured across the savepoint pair rather than read off the source.

    Also pins the two shapes the port depends on and the archive does not advertise: 'tke' is
    serialized as '(nvec, ke1, ntim)' and the savepoint reader squeezes the time axis away, so
    a capture with more than one TKE time level would silently hand this section the wrong
    profile; and 'imode_tkediff == 2' is what makes 'upd_prof' a TKE rather than a 'q', which
    is the difference between taking the square root and not.

    'imode_tkediff' is not serialized, so it is pinned indirectly, through 'TurbulenceConfig',
    which freezes it at 2 -- and through 'test_the_clamp_on_negative_tke_is_load_bearing'
    below, whose negative values only exist because the diffused quantity is an energy.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="9", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="10", date=date)

    assert utils.fields_that_changed(data_provider, before, after) == {"td_tketens"}
    assert entry.ntim() == 1, "'tke' carries more than one time level; 'tke()' squeezes it away."
    assert entry.ntur() == 1, "'ntur' is not the only serialized TKE time level."
    assert entry.fr_tke() == 1.0 / entry.dt_tke()


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_10_writes_exactly_these_rows(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Rows 1..nlev-1 change; the model top does not; the surface row cannot tell.

    The third claim is the reason this test exists. 'tketens(:,ke1)' is zero at both savepoints,
    so the Fortran's surface assignment is invisible to any comparison that starts from the
    entry state -- and that is exactly the kind of silently-vacuous row the copy-of-entry
    convention is meant to expose. It is asserted here so that a future capture in which the
    surface row arrives non-zero fails this test rather than quietly turning
    'test_compute_turbulent_velocity_scale_tendency_agrees_with_icon' into a real check of a
    row nobody looked at.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="9", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="10", date=date)
    nlev = entry.ke()
    columns = slice(before.ivstart(), before.ivend())

    on_entry = before.tketens().asnumpy()[columns]
    on_exit = after.tketens().asnumpy()[columns]
    rows_that_differ = tuple(np.flatnonzero((on_entry != on_exit).any(axis=0)).tolist())

    assert rows_that_differ == tuple(range(UPPERMOST_TENDENCY_LEVEL, nlev))
    assert not np.any(on_exit[:, nlev]), "the surface q tendency ICON returns is not zero."
    assert not np.any(on_entry[:, nlev]), (
        "the surface row of 'tketens' is no longer zero at the entry savepoint, so the "
        "whole-slab comparison now covers it; this test's premise has changed."
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_clamp_on_negative_tke_is_load_bearing(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The solve really does return negative TKE, so 'MAX(upd_prof, 0)' is not decoration.

    A statement about the reference rather than about the port: it establishes that dropping
    the clamp would produce NaN on this data and not merely a different rounding somewhere,
    which is what makes the comparison below a test of the clamp as well as of the arithmetic.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="9", date=date)
    nlev = entry.ke()
    columns = slice(before.ivstart(), before.ivend())

    written = before.upd_prof().asnumpy()[columns, UPPERMOST_TENDENCY_LEVEL:nlev]
    assert np.count_nonzero(written < 0.0) > 0


# --------------------------------------------------------------------------- the translation --


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_turbulent_velocity_scale_tendency_agrees_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The whole 'tketens' slab as 'turbdiff' leaves it, model top included.

    The output starts as a copy of the entry state, so the model-top row -- which the section
    must not write, and where the advection tendency the routine was handed still sits -- is
    part of the comparison rather than excluded from it. A vertical domain one row too high
    fails here.
    """
    tendency, after, columns = _run_the_tendency(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_turbulent_velocity_scale_tendency",
        "tketens [q tendency of the TKE diffusion]",
        tendency,
        after.tketens(),
        columns=columns,
    )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_surface_tendency_is_written_rather_than_left_alone(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The surface row is set to zero, not inherited -- which the reference alone cannot show.

    'test_section_10_writes_exactly_these_rows' establishes that the row is zero on both sides
    of the section, so the ordinary comparison passes with or without the Fortran's
    'tketens(:,ke1) = z0'. Poisoning that row of the output buffer with NaN removes the
    coincidence: the program must produce the zero itself, or the slab comes back with a NaN in
    it and the bit-exact comparison fails.
    """
    tendency, after, columns = _run_the_tendency(
        data_provider, date, backend, poison="output-surface"
    )

    utils.assert_agrees_with_icon(
        "compute_turbulent_velocity_scale_tendency",
        "tketens with the surface row of the output buffer poisoned",
        tendency,
        after.tketens(),
        columns=columns,
    )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_surface_row_of_the_updated_profile_is_not_read(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The boundary branch reads nothing, and only the branch a row selects is evaluated.

    Section 9) never writes the surface row of 'zaux(:,:,1)': it still holds section 0)'s Exner
    factor there, a plausible number near one whose square root would look like a tendency
    rather than like a defect. So the row is poisoned with NaN and the result must stay
    bit-exact. A backend that evaluated both branches everywhere and selected afterwards would
    return NaN at the surface; this is the same measurement section 1a) makes for its own
    boundary, and it is what keeps 'concat_where' honest as gt4py changes.
    """
    tendency, after, columns = _run_the_tendency(
        data_provider, date, backend, poison="input-surface"
    )

    utils.assert_agrees_with_icon(
        "compute_turbulent_velocity_scale_tendency",
        "tketens with the surface row of 'upd_prof' poisoned",
        tendency,
        after.tketens(),
        columns=columns,
    )
