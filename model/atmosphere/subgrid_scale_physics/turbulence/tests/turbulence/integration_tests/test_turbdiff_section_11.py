# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 11) of 'turbdiff', the interpolation of SDSS back to main levels.

'turbdiff-10-exit' supplies the inputs and 'turbdiff-exit' the expected outputs, for the four
timesteps 'exp.mch_icon-ch2_small' serializes. Section 11) is the last one, so its exit savepoint
is the routine's own exit: what these tests pin is not only a section boundary but the state
'turbdiff' hands back to the ICON interface.

WHAT THIS SECTION WRITES
------------------------
'rcld' on main levels 1..ke-1, and nothing else -- measured, not read off the source, by
'test_section_11_writes_only_the_supersaturation_deviation'. Within 'ivstart:ivend' every one of
those 79 rows differs from its input at every column and all four dates, so the copy-of-entry
convention gives full coverage here rather than a partly vacuous comparison.

WHY THE OUTPUT SET NEEDS A LOCAL HELPER
---------------------------------------
'utils.fields_that_changed' cannot be used on this savepoint pair, and section 11) is the only
section where that is so. It reads every name serialized at the entry savepoint from BOTH
savepoints, which holds for any two section savepoints -- they share one field table by
construction, being one Fortran hook called fifteen times. 'turbdiff-exit' is a different hook
with a different list: five names are section-only ('td_hor_scale', 'td_layr', 'td_lays',
'td_nvor', 'td_xri') and five are exit-only ('td_gz0', 'td_tvm', 'td_tvh', 'td_tkred_sfc',
'td_tkred_sfc_h'), so the helper raises 'SerialboxError: field does not exist at savepoint'.
'_fields_that_changed_across_the_exit_hook' below does the same comparison over the 28 names the
two hooks share, and the test closes both gaps rather than leaving them: the five exit-only names
are compared against 'turbdiff-entry' instead, and the five section-only ones are argued from the
30 lines of Fortran between the two hooks, which name no array but 'rcld'.

THE BOUNDARY ROW IS THE WHOLE OF THE TRANSLATION RISK
-----------------------------------------------------
The arithmetic is one addition and one exact halving; there is no operand order to get wrong and
nothing for a compiler flag to change. What can be wrong is the vertical structure, so that is
what the two remaining tests measure. The model top is a copy of the half level below it rather
than a mean with the half level above -- because that half level holds no SDSS -- and the two
lowest rows are not written at all. 'test_the_model_top_is_a_copy_rather_than_a_mean' shows the
data can tell those two coefficient choices apart at every column, and
'test_only_the_half_levels_between_the_boundaries_are_read' poisons the two rows the stencil must
never read and requires the result to stay bit-exact.

Selecting the top row's coefficients uses 'concat_where', which the embedded backend cannot
execute in gt4py 1.1.10 (see 'model/testing/filters.py'), so the two tests that run the program
carry 'uses_concat_where' and xfail there; the output-set test does not run it and covers all
three backends. Why the boundary is merged into one program instead of being a program of its own
is in the package README, section "Boundary rows".
"""

from __future__ import annotations

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.interpolate_supersaturation_deviation_to_main_levels import (
    interpolate_supersaturation_deviation_to_main_levels,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: The one storage slot section 11) writes, under the name it is serialized as. Asserted against
#: the data by 'test_section_11_writes_only_the_supersaturation_deviation'.
SECTION_11_OUTPUT_SLOTS = frozenset({"td_rcld"})

#: Serialized at the section hook but not at 'turbdiff-exit': 'turbdiff' locals and the iteration
#: counter, which never reach the interface and which the exit hook therefore does not carry.
#: Section 11) does not write them -- the Fortran between the two hooks names no array but 'rcld'
#: -- but that is the one claim here that is argued rather than measured, so it is named.
SECTION_ONLY_NAMES = frozenset({"td_hor_scale", "td_layr", "td_lays", "td_nvor", "td_xri"})

#: Serialized at 'turbdiff-exit' but not at the section hook: interface fields that 'turbdiff'
#: receives from 'turbtran' and passes through. They cannot be compared across the exit hook, so
#: 'test_section_11_writes_only_the_supersaturation_deviation' compares them across the whole
#: routine instead, where they are also unchanged.
EXIT_ONLY_NAMES = frozenset({"td_gz0", "td_tvm", "td_tvh", "td_tkred_sfc", "td_tkred_sfc_h"})


def _fields_that_changed_across_the_exit_hook(
    data_provider: sb.IconSerialDataProvider,
    before: sb.IconTurbdiffSectionSavepoint,
    after: sb.IconTurbulenceSavepoint,
) -> frozenset[str]:
    """'utils.fields_that_changed' over the names the two differently-shaped hooks share.

    Identical in what it asserts and why -- every name read from both savepoints and compared
    over 'ivstart:ivend', with NaN deliberately never equal to NaN -- and different only in
    skipping the names one of the two hooks does not serialize. See the module docstring for why
    that difference exists and how the skipped names are covered instead.
    """
    ivstart, ivend = before.ivstart(), before.ivend()
    window = slice(ivstart, ivend)
    shared = set(data_provider.serializer.fields_at_savepoint(after.savepoint))
    changed = set()
    for name in data_provider.serializer.fields_at_savepoint(before.savepoint):
        if name not in shared:
            continue
        entry = np.asarray(data_provider.serializer.read(name, before.savepoint))
        exit_ = np.asarray(data_provider.serializer.read(name, after.savepoint))
        masked = window if entry.ndim >= 2 and entry.shape[0] > ivend else slice(None)
        if not np.array_equal(entry[masked], exit_[masked]):
            changed.add(name)
    return frozenset(changed)


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_11_writes_only_the_supersaturation_deviation(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Section 11) changes 'rcld' and nothing else, and the routine ends there.

    Three claims, in the order they are made: the shared names differ in 'td_rcld' alone; the
    two hooks differ in exactly the ten names the module docstring accounts for, so a future
    recapture that serializes something new here fails rather than being silently skipped; and
    the five names that only 'turbdiff-exit' carries are unchanged across the whole of 'turbdiff',
    which is a stronger statement than the one section 11) needs.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="10", date=date)
    after = data_provider.from_savepoint_turbdiff_exit(date=date)

    assert (
        _fields_that_changed_across_the_exit_hook(data_provider, before, after)
        == SECTION_11_OUTPUT_SLOTS
    )

    serializer = data_provider.serializer
    at_section = frozenset(serializer.fields_at_savepoint(before.savepoint))
    at_exit = frozenset(serializer.fields_at_savepoint(after.savepoint))
    assert at_section - at_exit == SECTION_ONLY_NAMES
    assert at_exit - at_section == EXIT_ONLY_NAMES

    columns = slice(entry.ivstart(), entry.ivend())
    for name in sorted(EXIT_ONLY_NAMES):
        assert np.array_equal(
            np.asarray(serializer.read(name, entry.savepoint))[columns],
            np.asarray(serializer.read(name, after.savepoint))[columns],
        ), f"'{name}' is not a pass-through of 'turbdiff' after all; section 11) may write it."


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_interpolate_supersaturation_deviation_to_main_levels(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """One program reproduces the whole main-level 'rcld' profile 'turbdiff' returns.

    The output is allocated as a COPY OF THE ENTRY STATE and compared over the whole column, so
    the comparison covers the two rows the section does not write as well as the 79 it does: the
    lowest main level keeping its half-level value, and the surface row being left alone, are
    assertions here rather than omissions.

    Inputs and outputs are separate fields, which the Fortran's are not. That is what makes the
    claim in the stencil's docstring -- that its sequential k-loop carries nothing -- something
    this test can fail on rather than something the translation assumes.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="10", date=date)
    after = data_provider.from_savepoint_turbdiff_exit(date=date)
    nlev = entry.ke()
    columns = slice(before.ivstart(), before.ivend())

    on_half_levels = before.sdss()
    on_main_levels = utils.copy_of(on_half_levels, backend)

    interpolate_supersaturation_deviation_to_main_levels.with_backend(backend)(
        supersaturation_deviation_on_half_levels=on_half_levels,
        supersaturation_deviation_on_main_levels=on_main_levels,
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(nlev - 1),
        offset_provider={dims.Koff.value: dims.KDim},
    )

    utils.assert_agrees_with_icon(
        "interpolate_supersaturation_deviation_to_main_levels",
        "rcld",
        on_main_levels,
        after.rcld(),
        columns=columns,
        levels=slice(0, nlev + 1),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_model_top_is_a_copy_rather_than_a_mean(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """ICON copies the second half level into the top main level, and the data says which.

    A statement about the reference, not about the port: it establishes that the boundary the
    stencil selects for is one this capture can actually distinguish, so that the comparison in
    'test_interpolate_supersaturation_deviation_to_main_levels' is not silently passing on a
    degenerate column. Averaging the top main level would halve it, because the half level above
    holds exactly zero -- 'solve_turb_budgets' starts at the second half level and never writes
    the first -- which is the reason the Fortran writes that row as a copy in a loop of its own.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="10", date=date)
    after = data_provider.from_savepoint_turbdiff_exit(date=date)
    columns = slice(before.ivstart(), before.ivend())

    on_half_levels = before.sdss().asnumpy()[columns]
    on_main_levels = after.rcld().asnumpy()[columns]

    assert np.array_equal(on_main_levels[:, 0], on_half_levels[:, 1]), (
        "the top main level of 'rcld' is not a copy of the half level below it."
    )
    assert not np.any(on_half_levels[:, 0]), (
        "the topmost half level of 'rcld' is not zero; 'solve_turb_budgets' appears to write it "
        "in this capture, which is the premise the copy at the model top rests on."
    )
    mean = (on_half_levels[:, 0] + on_half_levels[:, 1]) * 0.5
    assert not np.any(on_main_levels[:, 0] == mean), (
        "the top main level agrees with the two-point mean at some column, so this data cannot "
        "tell the boundary coefficients apart there."
    )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_only_the_half_levels_between_the_boundaries_are_read(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """Neither the topmost half level nor the surface one reaches the result.

    Both are rows the section must not read: the model top because the row it feeds is a copy of
    its neighbour and not a mean with it, the surface because the averaging loop stops one main
    level short of it. Whether 'concat_where' evaluates only the branch a row selects is a
    property of GT4Py's domain inference rather than of the expression, so it is measured: both
    rows are poisoned with NaN and the whole profile must stay bit-exact. A backend that computed
    both branches everywhere and selected afterwards would return NaN at the model top; one whose
    vertical domain ran one row too far would return NaN at the lowest written main level.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="10", date=date)
    after = data_provider.from_savepoint_turbdiff_exit(date=date)
    nlev = entry.ke()
    columns = slice(before.ivstart(), before.ivend())

    poisoned = before.sdss().asnumpy().copy()
    poisoned[:, 0] = np.nan
    poisoned[:, nlev] = np.nan
    on_half_levels = gtx.as_field(before.sdss().domain, poisoned, allocator=backend)
    on_main_levels = utils.copy_of(before.sdss(), backend)

    interpolate_supersaturation_deviation_to_main_levels.with_backend(backend)(
        supersaturation_deviation_on_half_levels=on_half_levels,
        supersaturation_deviation_on_main_levels=on_main_levels,
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(nlev - 1),
        offset_provider={dims.Koff.value: dims.KDim},
    )

    utils.assert_agrees_with_icon(
        "interpolate_supersaturation_deviation_to_main_levels",
        "rcld with the two unread half levels poisoned",
        on_main_levels,
        after.rcld(),
        columns=columns,
        levels=slice(0, nlev + 1),
    )
