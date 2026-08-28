# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 8) of 'turbdiff': the circulation term as an extra TKE flux density.

The oracle is a real ICON run: 'turbdiff-7-exit' supplies the inputs and 'turbdiff-8-exit' the
expected outputs, for the four timesteps that 'exp.mch_icon-ch2_small' serializes.

WHAT THIS SECTION WRITES
------------------------
One quantity in one storage, measured and not read off the source
('test_section_8_writes_only_the_scratch_array'):

    'hlp'   cur_prof   the virtual TKE profile carrying the circulation term   k = 2..ke1

'hlp' is the scheme's general-purpose scratch and has no stable meaning across sections; here it
is 'cur_prof'. Everything else the hook serializes -- 'frm', 'zaux', 'zvari', 'tke', 'tkvm',
'tkvh' -- is byte-identical across the two savepoints, so this section is one program.

THIS SECTION IS A GENUINE VERTICAL RECURRENCE
---------------------------------------------
'cur_prof(k) = (cur_prof(k-1) - sav_prof(k-1))*fakt + wert + sav_prof(k)' reads the value the
previous 'k' iteration of the same loop wrote, which is the test the package README states for
deciding whether a '!$ACC LOOP SEQ' has to become a 'scan_operator'. It does, and it is the only
statement in this section: one program, one scan.

Two things about that scan are easy to get wrong and invisible in the result, so each has a test
of its own rather than a comment. 'test_the_recurrence_runs_downward_from_the_second_half_level'
pins the direction and the seed -- accumulating upward from the surface, or dropping the carry
altogether, both give a smooth-looking field that is not ICON's. And
'test_the_c_diff_limit_correction_is_unity_in_this_capture' recovers 'fakt' from the data,
because it is a namelist quantity the port would otherwise be asserting its own assumption about.

WHAT THIS CAPTURE DOES NOT COVER
--------------------------------
  * 'fakt != 1'. 'c_diff = 0.2' and 'epsi = 1e-6', so 'c_diff_llim = MAX(epsi, c_diff)' is
    'c_diff' and the correction factor is exactly one. The branch of the Fortran that the factor
    exists for -- 'c_diff' at or below 'epsi', where section 6) has to limit 'expl_mom' to avoid
    the division by zero this section performs -- is not exercised by any of the four dates.
  * The 'ELSEIF (ldotkedif)' branch at :2390, where 'cur_prof' is aliased to 'sav_prof' and this
    program is not called at all. 'lcircterm' is true here (the section writes 'hlp', which only
    that branch does), so the alternative has no oracle either. It needs no stencil: it is a
    pointer assignment, and what it implies for the caller is in the program docstring.
  * 'itndcon = 0'. A host scalar telling section 9) not to add a separate tendency, set
    identically in both branches. Nothing to port and nothing serialized.

Nothing here selects a boundary row by a coefficient -- the uppermost row is distinguished by a
flag carried in the scan, as section 9)'s forward elimination does it -- so no 'concat_where' is
involved and the one stencil is validated on 'embedded' as well as on the compiled backends.

THESE TESTS ARE NOT MARKED 'embedded_too_slow', AND THAT IS A DECISION
---------------------------------------------------------------------
The embedded backend runs a 'scan_operator' as a Python loop over 'nlev' rows, which costs 170 s
per date here against 0.4 s on 'gtfn_cpu' and 'dace_cpu' -- 11 minutes for the module. That is
slow, but it is not the ~8 minutes PER TEST that got section 9)'s scans marked, and embedded is
the only backend whose result is not produced by a compiler this port also has to trust. The
eleven minutes buy an independent check of the carry, so they are paid. Re-measure before
copying the decision: it scales with 'nlev' and with the number of computed columns.
"""

from __future__ import annotations

from typing import NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import turbulence
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_virtual_tke_profile import (
    compute_virtual_tke_profile,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: The offset provider the scan's shifted input needs: it reads the saved profile one half level
#: up.
_KOFF = {dims.Koff.value: dims.KDim}

#: Zero-based row of the uppermost half level this section writes, Fortran 'cur_prof(:,2)'. The
#: model top is not written: 'turbdiff' has no TKE-diffusion level there, and section 6) did not
#: write the saved profile there either.
UPPERMOST_HALF_LEVEL = 1


def _tke_diffusion_limit_correction(config: turbulence.TurbulenceConfig) -> float:
    """'fakt' of turb_diffusion.f90:2361, 'c_diff / c_diff_llim'; one scalar per run.

    'c_diff_llim' is the low-limited length-scale factor section 6) built 'expl_mom' with
    (:2135-2139), and it is limited only while the raw circulation term is active, because this
    section is the only place that divides by 'expl_mom'. The quotient undoes the limit again for
    the pure TKE diffusion.

    A host-side quantity, like section 6)'s 'c_diff_llim' itself: it depends on the namelist
    alone. Recomputed here rather than imported from that section's test module, so that this
    module states the whole of what it assumes.
    """
    circulation_term_is_active = config.pat_len > 0.0 and config.ltkenst
    low_limited = max(config.epsi, config.c_diff) if circulation_term_is_active else config.c_diff
    return config.c_diff / low_limited


#: The correction factor the reference run used, from the compiled-in defaults
#: ('c_diff = 0.20', 'epsi = 1e-6'; mo_turbdiff_config.f90:240, :151).
#: 'test_the_c_diff_limit_correction_is_unity_in_this_capture' recovers it from the data.
TKE_DIFFUSION_LIMIT_CORRECTION = _tke_diffusion_limit_correction(turbulence.TurbulenceConfig())


class Section8(NamedTuple):
    """One timestep of section 8): the savepoints, the bounds and the single computed output."""

    entry: sb.IconTurbdiffEntrySavepoint
    before: sb.IconTurbdiffSectionSavepoint
    after: sb.IconTurbdiffSectionSavepoint
    ke1: int
    #: Half-open range of columns 'turbdiff' computed; the rest of the slab is untouched memory
    #: holding plausible values, so every comparison below is masked with it.
    columns: slice
    virtual_tke_profile: gtx.Field


def _run_section_8(data_provider, date: str, backend) -> Section8:
    """Run the one program of section 8) on the 'turbdiff-7-exit' state of one timestep.

    The output starts as the entry state of the storage it lands in, per the convention in
    'utils': the rows the section does not write -- here the model top alone -- are then asserted
    untouched by the same whole-slab comparison rather than excluded from it.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="7", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="8", date=date)
    ke1 = entry.ke1()

    virtual_tke_profile = utils.copy_of(before.hlp(), backend)
    compute_virtual_tke_profile.with_backend(backend)(
        saved_tke_profile=before.sav_prof(),
        cke_flux_at_main_levels=before.cke_flux_at_main_levels(),
        explicit_diffusion_momentum=before.expl_mom(),
        tke_diffusion_limit_correction=TKE_DIFFUSION_LIMIT_CORRECTION,
        virtual_tke_profile=virtual_tke_profile,
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        vertical_start=gtx.int32(UPPERMOST_HALF_LEVEL),
        vertical_end=gtx.int32(ke1),
        offset_provider=_KOFF,
    )

    return Section8(
        entry=entry,
        before=before,
        after=after,
        ke1=ke1,
        columns=slice(before.ivstart(), before.ivend()),
        virtual_tke_profile=virtual_tke_profile,
    )


# ------------------------------------------------------------------- what the section writes --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_8_writes_only_the_scratch_array(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The section's true output set, measured rather than read off the source.

    That it is exactly 'td_hlp' also settles which branch of the section ran: only the
    'lcircterm' branch writes anything at all, so 'lcircterm' was true. 'lcircterm' is
    'pat_len > 0 .AND. ltkenst' (turb_diffusion.f90:945, :959), neither of which is serialized,
    so this is the only evidence available for it.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="7", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="8", date=date)

    assert utils.fields_that_changed(data_provider, before, after) == {"td_hlp"}


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_8_writes_exactly_these_rows(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Every half level below the model top, and the model top itself untouched.

    The vertical extent is where a translation of this section can go wrong most quietly: the
    accumulation is smooth, so a scan that starts one row too low or stops one row too high
    produces a field that looks entirely plausible. Measuring the range from the two savepoints
    is what makes the whole-slab comparison below able to see that.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="7", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="8", date=date)
    columns = slice(before.ivstart(), before.ivend())

    differs = before.hlp().asnumpy()[columns] != after.hlp().asnumpy()[columns]
    rows = tuple(np.flatnonzero(differs.any(axis=0)).tolist())

    assert rows == tuple(range(UPPERMOST_HALF_LEVEL, entry.ke1()))


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_c_diff_limit_correction_is_unity_in_this_capture(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'fakt' recovered from the data, because it is a namelist quantity and not a serialized one.

    Inverting the recurrence for the one unknown,

        fakt = (cur_prof(k) - frm(k)/expl_mom(k) - sav_prof(k)) / (cur_prof(k-1) - sav_prof(k-1))

    gives it at every level of every computed column. The recovery is arithmetic on already
    rounded values and the denominator is a difference of two nearby numbers, so it agrees with
    1.0 to about 1e-11 rather than exactly; the point is that the number is 'c_diff/c_diff_llim'
    with the limit not binding, and not something else.

    THE CAPTURE THEREFORE DOES NOT COVER 'fakt != 1'. Whatever the port does with the factor,
    only its identity is validated here. A configuration with 'c_diff <= epsi' would exercise it,
    and none of the four dates is one.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="7", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="8", date=date)
    columns = slice(before.ivstart(), before.ivend())

    saved = before.sav_prof().asnumpy()[columns]
    virtual = after.hlp().asnumpy()[columns]
    flux_density = before.cke_flux_at_main_levels().asnumpy()[columns]
    diffusion_momentum = before.expl_mom().asnumpy()[columns]
    increment = flux_density / diffusion_momentum

    # The uppermost row seeds the recurrence rather than continuing it, so the quotient is taken
    # from the second one down; a vanishing carry would make it 0/0 and is dropped.
    carried = (virtual - saved)[:, UPPERMOST_HALF_LEVEL:-1]
    remainder = (virtual - increment - saved)[:, UPPERMOST_HALF_LEVEL + 1 :]
    resolvable = carried != 0.0
    recovered = remainder[resolvable] / carried[resolvable]

    assert resolvable.sum() > 0
    np.testing.assert_allclose(recovered, TKE_DIFFUSION_LIMIT_CORRECTION, rtol=1e-9, atol=0.0)

    config = turbulence.TurbulenceConfig()
    assert config.c_diff > config.epsi
    assert TKE_DIFFUSION_LIMIT_CORRECTION == 1.0


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_recurrence_runs_downward_from_the_second_half_level(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Direction, seed and carry, each shown to matter, in numpy and without any backend.

    A wrong direction or a dropped carry does not produce a broken-looking field -- the
    accumulation is smooth either way -- so 'the stencil agrees with ICON' is the only thing that
    would catch it, and only after the stencil exists. This states it in terms of the reference
    data alone, so that the four dates say what the scan has to be before anything is written:

      * accumulating downward from 'cur_prof(:,2) = sav_prof(:,2)' reproduces ICON BIT FOR BIT,
        which also fixes the operand order of the statement;
      * dropping the carry ('cur_prof(k) = wert + sav_prof(k)') does not;
      * accumulating upward from the surface row instead does not.

    The reproduction being bit-exact and not merely close is what makes 'Exact()' the right gate
    for the stencil: any disagreement there is the backend's re-association, not the arithmetic.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="7", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="8", date=date)
    ke1 = entry.ke1()
    columns = slice(before.ivstart(), before.ivend())
    fakt = TKE_DIFFUSION_LIMIT_CORRECTION

    saved = before.sav_prof().asnumpy()[columns]
    reference = after.hlp().asnumpy()[columns]
    flux_density = before.cke_flux_at_main_levels().asnumpy()[columns]
    diffusion_momentum = before.expl_mom().asnumpy()[columns]
    increment = flux_density / diffusion_momentum

    downward = before.hlp().asnumpy()[columns].copy()
    downward[:, UPPERMOST_HALF_LEVEL] = saved[:, UPPERMOST_HALF_LEVEL]
    for level in range(UPPERMOST_HALF_LEVEL + 1, ke1):
        downward[:, level] = (
            (downward[:, level - 1] - saved[:, level - 1]) * fakt
            + increment[:, level]
            + saved[:, level]
        )
    assert np.array_equal(downward, reference)

    without_carry = downward.copy()
    without_carry[:, UPPERMOST_HALF_LEVEL + 1 :] = (
        increment[:, UPPERMOST_HALF_LEVEL + 1 :] + saved[:, UPPERMOST_HALF_LEVEL + 1 :]
    )
    assert not np.array_equal(without_carry, reference)

    upward = before.hlp().asnumpy()[columns].copy()
    upward[:, ke1 - 1] = saved[:, ke1 - 1]
    for level in range(ke1 - 2, UPPERMOST_HALF_LEVEL - 1, -1):
        upward[:, level] = (
            (upward[:, level + 1] - saved[:, level + 1]) * fakt
            + increment[:, level]
            + saved[:, level]
        )
    assert not np.array_equal(upward, reference)


# ------------------------------------------------------------------------- the virtual profile --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_virtual_tke_profile_agrees_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The section's whole product, over the whole slab.

    The comparison includes the model top, which this section must not write and which still
    holds whatever section 2a) left in the scratch: a scan that ran one row too far would
    overwrite it, and the buffer started as the entry state precisely so that this shows up as a
    failure rather than as nothing.

    A recurrence is where a translation most easily loses bit-exactness -- 80 levels of carried
    state, each one an opportunity for the backend to accumulate in a different order than the
    sequential Fortran loop -- so this is worth reading as a measurement and not only as a pass.
    """
    run = _run_section_8(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_virtual_tke_profile",
        "hlp [cur_prof]",
        run.virtual_tke_profile,
        run.after.hlp(),
        columns=run.columns,
    )
