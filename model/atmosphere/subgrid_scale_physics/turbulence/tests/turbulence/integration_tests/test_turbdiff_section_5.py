# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 5) of 'turbdiff', the temperature tendencies of the TKE sources.

Section 5) is turb_diffusion.f90:2053-2117 between 'turbdiff-4-exit' and 'turbdiff-5-exit'. It is
the second section of the scheme, after 2b), that **does nothing at all in the operational
configuration**, and this module is what establishes that rather than a port of it. There are no
stencils and no entries in 'gate_registry.GATES' for section 5).

WHAT THE DATA SAYS
------------------
'turbdiff-5-exit' is a byte-for-byte copy of 'turbdiff-4-exit'. All 33 serialized fields are
identical over the whole 'nproma' slab -- not merely over 'ivstart:ivend' -- at all four dates
('test_section_5_writes_nothing_in_this_capture'). 'utils.fields_that_changed' returns the empty
set for the pair.

WHY, AND WHAT WOULD HAVE TO CHANGE
----------------------------------
The section is two guarded blocks and nothing else, and both guards are false:

  * 'IF (tdc%ltmpcor)' (:2057-2100) -- the dissipative-heating and phase-diffusion tendency. It
    fills 'hlp' with a length-scale-scaled temperature tendency on half levels 2..ke1 and then
    ACCUMULATES it into 't_tens' on main levels 1..ke.
  * 'IF (ldocirflx)' (:2102-2112), where 'ldocirflx = tdc%lcirflx .AND. lcircterm' (:2047) --
    'hlp = ABS(frh*tkvh)', the magnitude of the updated buoyant TKE source, saved for the
    circulation heat flux.

'ldocirflx' is serialized from section 4) on and reads False at every savepoint and every date;
'test_the_circulation_heat_flux_is_switched_off' asserts it at both ends of this section.
'ltmpcor' is not serialized, so it is recovered from the data instead
('test_the_enthalpy_budget_correction_is_switched_off'): 'solve_turb_budgets' fills 'ediss' only
under 'lpres_edr .OR. ltmpcor' (turb_utilities.f90:1830), and the 'ediss' storage is untouched
across every section of this run -- it still holds uninitialised memory, three quarters of it
negative, which the positive-definite 'q**3/(d_m*l)' of turb_utilities.f90:1846 cannot produce.
An untouched 'ediss' therefore means both flags are false. It also means the section could not
have run meaningfully even if 'ltmpcor' alone were set, since ':2067' reads that same 'ediss'.

The namelist agrees, which is the check and not the argument: 'exp.mch_icon-ch2_small' echoes
'LTMPCOR = F' and 'LCPFLUC = F' in 'NAMELIST_ICON_output_atm', and 'lcirflx' is not a
'turbdiff_nml' variable at all and keeps its '.FALSE.' default (mo_turbdiff_config.f90:281).

THE PORT CANNOT REACH THIS SECTION EITHER
-----------------------------------------
'ltmpcor', 'lcpfluc' and 'lcirflx' are all three on 'turbulence.FROZEN_SWITCHES' at '.FALSE.', so
a 'TurbulenceConfig' that would activate section 5) is refused with a 'NotImplementedError'
before anything runs ('test_the_port_refuses_a_configuration_that_would_run_section_5'). Porting
the section would therefore add unreachable code with no oracle to validate it against, which is
what the port spec's rule on uncovered branches (5.4) tells us not to do. The four tests below
are the canary instead: if a recapture turns either guard on, they fail and say so.

WHAT A LATER PORT OF THIS SECTION WOULD HAVE TO GET RIGHT
---------------------------------------------------------
Recorded here because reading it off the Fortran again is the expensive part, and because none
of it is visible in a capture where the section is dead:

  * 't_tens' is ACCUMULATED into, not overwritten: ':2085' and ':2094' both read the incoming
    tendency. A port must start from the entry-savepoint value.
  * The two 't_tens' loops are one field with two vertical cases, so the boundary-row rule of
    the package README selects 'concat_where': the top main level takes 'hlp(2)/(l(1)+l(2))' --
    'hlp(1)' is deliberately absent, the 'hlp' loop starting at k=2 -- and levels 2..ke take
    '(hlp(k)+hlp(k+1))/(l(k)+l(k+1))'.
  * 'hlp' is written on half levels 2..ke1 inclusive; the comment at ':2074-2077' says the
    surface row exists only so the main-level interpolation below it has something to read.
  * At section 5) the reused storages mean: 'len_scale' is 'mixing_length()', 'zaux(:,:,2)' is
    'r_cpd()', 'zvari(:,:,3)' and 'zvari(:,:,4)' (Fortran 'vap' = 4 and 'liq' = 5) are
    'effective_gradient()', and 'ediss' is 'edr()', which this capture never defines.
"""

from __future__ import annotations

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import turbulence
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: The number of fields the section hook serializes from section 4) on: the 31 of sections
#: 0)..3) plus 'td_ldoexpcor' and 'td_ldocirflx'. Asserted so that a recapture which instruments
#: something new here fails instead of being silently skipped by a comparison that never sees it.
SERIALIZED_FIELD_COUNT = 33

#: The switches that would make section 5) do something, and the value each is frozen at.
#: 'ldocirflx' is 'lcirflx .AND. lcircterm', so freezing 'lcirflx' is enough for the second block;
#: 'lcpfluc' only selects which of the two 'ltmpcor' formulations runs, and is listed because a
#: port of the section would need it too.
SWITCHES_THAT_WOULD_ACTIVATE_SECTION_5 = ("ltmpcor", "lcpfluc", "lcirflx")


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_5_writes_nothing_in_this_capture(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'turbdiff-5-exit' is a copy of 'turbdiff-4-exit', field for field and byte for byte.

    Two statements, the second stronger than the section needs. First the standard one: nothing
    changes over the computed columns, which is what 'utils.fields_that_changed' measures and
    what bounds the stencils a section may contain -- here, none. Then the same comparison
    unmasked, over the whole 'nproma' slab, which also rules out a write outside 'ivstart:ivend'
    that the masked comparison could not see.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="4", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="5", date=date)

    assert utils.fields_that_changed(data_provider, before, after) == frozenset()

    serializer = data_provider.serializer
    names = sorted(serializer.fields_at_savepoint(before.savepoint))
    assert len(names) == SERIALIZED_FIELD_COUNT
    differing = [
        name
        for name in names
        if not np.array_equal(
            np.asarray(serializer.read(name, before.savepoint)),
            np.asarray(serializer.read(name, after.savepoint)),
        )
    ]
    assert differing == [], (
        f"section 5) changed {differing} in this capture, so one of its two guards is no longer "
        "false and the section needs porting after all."
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_circulation_heat_flux_is_switched_off(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'ldocirflx' is false at both ends of section 5), so ':2102-2112' does not run.

    The one guard of the section that the capture states outright: section 4) assigns
    'ldocirflx = lcirflx .AND. lcircterm' and the hook serializes it from there on.
    """
    for section in ("4", "5"):
        savepoint = data_provider.from_savepoint_turbdiff_section(section=section, date=date)
        assert savepoint.ldocirflx() is False


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_enthalpy_budget_correction_is_switched_off(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'ltmpcor' is false, recovered from 'ediss' because the flag itself is not serialized.

    'solve_turb_budgets' writes the dissipation rate into 'ediss' only under
    'lpres_edr .OR. ltmpcor' (turb_utilities.f90:1830), as 'q**3/(d_m*l)' (:1846), which is
    positive definite. Here the storage is identical at every section savepoint and holds
    negative values at most of its points, so it was never written and both flags are false.

    That is also why the first block of section 5) could not have produced anything usable even
    if 'ltmpcor' alone were set: ':2067' reads this same 'ediss'.
    """
    serializer = data_provider.serializer
    savepoints = [
        data_provider.from_savepoint_turbdiff_section(section=section, date=date)
        for section in sb.TURBDIFF_SECTIONS[: sb.TURBDIFF_SECTIONS.index("5") + 1]
    ]
    dissipation = [np.asarray(serializer.read("td_edr", sp.savepoint)) for sp in savepoints]

    for savepoint, values in zip(savepoints[1:], dissipation[1:], strict=True):
        assert np.array_equal(dissipation[0], values), (
            f"'ediss' changed by 'turbdiff-{savepoint.section}-exit'; 'solve_turb_budgets' now "
            "writes it, so 'lpres_edr' or 'ltmpcor' is on and section 5) may no longer be dead."
        )
    assert np.any(dissipation[0] < 0.0), (
        "'ediss' is non-negative everywhere, which is what 'q**3/(d_m*l)' would give: it may "
        "have been written after all, and 'ltmpcor' can no longer be ruled out this way."
    )


def test_the_port_refuses_a_configuration_that_would_run_section_5() -> None:
    """A 'TurbulenceConfig' that activates either block of section 5) raises.

    The other half of the argument for not porting the section: it is dead in this capture, and
    it is unreachable in the port for as long as these three switches stay on
    'FROZEN_SWITCHES'. If one of them is unfrozen, this fails and section 5) becomes a port task
    again rather than quietly becoming untested behaviour.
    """
    frozen = {switch.name: switch for switch in turbulence.FROZEN_SWITCHES}
    for name in SWITCHES_THAT_WOULD_ACTIVATE_SECTION_5:
        assert name in frozen, f"'{name}' is no longer frozen; section 5) may now be reachable."
        assert frozen[name].supported_value is False
        with pytest.raises(NotImplementedError, match=name):
            turbulence.TurbulenceConfig(**{name: True})
