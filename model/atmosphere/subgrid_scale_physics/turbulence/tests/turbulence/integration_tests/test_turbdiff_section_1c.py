# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 1c) of 'turbdiff', the initialisation of 'tke', 'tkvh' and 'tkvm'.

'turbdiff-1b-exit' supplies the inputs and 'turbdiff-1c-exit' the expected outputs, for the four
timesteps 'exp.mch_icon-ch2_small' serializes.

THERE IS NOTHING TO PORT: THE SECTION IS A MEASURED NO-OP HERE
--------------------------------------------------------------
Section 1c) (turb_diffusion.f90:1244-1319) is two guarded blocks and nothing else, and neither
guard is taken in this capture. All 31 fields serialized at the two savepoints are byte-identical
at all four dates -- not merely over 'ivstart:ivend' but over the whole 'nproma' slab -- so
'turbdiff-1c-exit' is a copy of 'turbdiff-1b-exit'. That is what
'test_section_1c_writes_nothing' asserts, and it is the only thing about this section that the
reference data can decide.

The three tests after it say WHY each guard is closed, so that a capture in which one of them
opens fails here rather than silently invalidating the claim above. This follows section 5.4 of
the port spec: a branch the data cannot cover is named, not hoped away. Section 2b) is handled the
same way, by 'test_turbdiff_2b_is_dead_in_this_configuration' in
'test_turbdiff_section_savepoints.py'.

WHAT THE TWO GUARDS ARE, AND HOW THEY DIFFER FROM EACH OTHER
-------------------------------------------------------------
'IF (lini)' at :1249 is NOT dead code. It is the scheme's cold start -- 'init_basic_atmo_turb'
for a first estimate of 'tkvm', 'tkvh' and 'tke' from a simplified TKE equilibrium, then
'tke(:,1,1) = tke(:,2,1)' for a vanishing TKE flux at the model top, then a fill of the remaining
'tke' time levels -- and the operational model executes it, once. 'turb_setup' sets
'lini = (iini > 0)' (turb_utilities.f90:364-388), and this capture serializes the last four of six
timesteps, all of them with 'iini = 0'. So the block is live in ICON and UNCOVERED here, which is
the weaker and more awkward of the two statements: porting it means porting
'init_basic_atmo_turb' with no oracle at all.

'IF (ltkeadapt)' at :1290 is off by namelist. 'ltkeadapt = (tdc%imode_tkemini == 2)' (:967) and
'imode_tkemini' defaults to 1 (mo_turbdiff_config.f90:372), which MCH does not override -- see
'test_the_prandtl_number_adaptation_is_switched_off' for how the capture proves that without the
namelist. Its body is the single statement 'tkvm = tprn*tkvh' over 'k=1..kem'.

WHERE 'init_basic_atmo_turb' HAS TO COME FROM
----------------------------------------------
Not from here. 'turbtran' calls it twice, at turb_transfer.f90:960 and :1501, and BOTH of
those calls are under 'lini' too -- 'IF (lini)' at :938 and
'IF (it_durch == it_start .AND. lini)' at :1498. So the routine is unreachable everywhere in
this capture and no savepoint pair in the archive constrains it. Porting the cold start is
therefore a single piece of work spanning 'turbtran' and this section, and it needs a capture
taken with 'iini > 0' before any of it can be validated. Writing it against this data would be
writing it untested.
"""

from __future__ import annotations

import numpy as np
import pytest

from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: The number of fields 'serialize_turbdiff_section' writes at a section boundary before
#: 'ldoexpcor'/'ldocirflx' exist, i.e. for sections 0)..3). Asserted alongside the no-op claim so
#: that a hook which stopped serializing something cannot make the section look inert.
FIELDS_AT_A_SECTION_SAVEPOINT = 31


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_1c_writes_nothing(date: str, *, data_provider: sb.IconSerialDataProvider) -> None:
    """'turbdiff-1c-exit' is a byte-for-byte copy of 'turbdiff-1b-exit'.

    Asserted twice over. 'utils.fields_that_changed' is the shared way a section's output set is
    established, and it reports the empty set; the loop below repeats the comparison over the
    full 'nproma' slab rather than the computed columns, which is the stronger statement and the
    one that holds here -- the section did not write outside its window either, because it did
    not write at all.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="1b", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="1c", date=date)

    assert utils.fields_that_changed(data_provider, before, after) == frozenset()

    names = sorted(data_provider.serializer.fields_at_savepoint(before.savepoint))
    assert len(names) == FIELDS_AT_A_SECTION_SAVEPOINT
    differing = [
        name
        for name in names
        if not np.array_equal(
            np.asarray(data_provider.serializer.read(name, before.savepoint)),
            np.asarray(data_provider.serializer.read(name, after.savepoint)),
        )
    ]
    assert differing == []


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_cold_start_is_not_reached_in_this_capture(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'lini' is false at every serialized timestep, so the whole 'IF (lini)' block is skipped.

    A coverage canary, not a property of the scheme: the block runs in the operational model,
    just not in the window this capture covers. 'turb_setup' derives 'lini = (iini > 0)'
    (turb_utilities.f90:364-388) and both are serialized at 'turbdiff-entry', so the two are
    asserted together -- 'lini' for what the branch reads and 'iini' for why it is what it is.

    If this ever fails, the archive contains an initialisation call, 'init_basic_atmo_turb' has
    acquired an oracle, and section 1c) stops being a no-op.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)

    assert entry.iini() == 0
    assert entry.lini() is False


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_tke_time_levels_are_a_single_slot(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'ntim == 1', so the 'DO n=2,ntim' fill of the cold start is an empty loop regardless.

    The NWP interface carries one TKE time level, which is why the reader squeezes the trailing
    axis away ('IconTurbdiffSectionSavepoint.tke'). Worth pinning separately from 'lini': it
    bounds how much of the cold-start block a future capture with 'iini > 0' would actually
    exercise, and a port that reproduced the time-level fill would be reproducing dead code as
    long as this holds.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)

    assert entry.ntim() == 1
    assert entry.ntur() == 1
    assert entry.nprv() == 1


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_prandtl_number_adaptation_is_switched_off(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'ltkeadapt' is false, and the capture proves it without the namelist being serialized.

    'ltkeadapt = (tdc%imode_tkemini == 2)' (turb_diffusion.f90:967) and neither 'imode_tkemini'
    nor the namelist is in the archive. What is in the archive is the SHAPE of 'tprn':
    mo_nwp_phy_state.f90:4705-4711 allocates 'prm_diag%tprn' with the full half-level shape when
    "imode_tkemini == 2 .OR. rsur_sher > 0" and with the '(1, 1, kblks)' dummy shape otherwise.
    'td_tprn' is '(1, 1)' here, which excludes 'imode_tkemini == 2' and closes the guard.

    That is also why the branch could not be validated even if it did run: its only input is a
    field this configuration never allocated.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="1b", date=date)
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)

    tprn = np.asarray(data_provider.serializer.read("td_tprn", before.savepoint))

    assert tprn.shape == (1, 1)
    assert tprn.shape != (entry.nvec(), entry.ke1())
