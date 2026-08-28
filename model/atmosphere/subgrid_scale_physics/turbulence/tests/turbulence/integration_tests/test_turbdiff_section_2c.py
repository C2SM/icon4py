# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 2c) of 'turbdiff', the final preparations before the turbulent budgets.

The oracle is a real ICON run: 'turbdiff-2b-exit' supplies the inputs and 'turbdiff-2c-exit' the
expected outputs, for the four timesteps 'exp.mch_icon-ch2_small' serializes. Section 2b) is
measured dead (the section map: zero of 87 fields differ across its savepoints), so the entry
state here is what section 2a) produced.

WHAT THIS SECTION WRITES
------------------------
'tkvm' and 'tkvh' at half levels 2..kem, and nothing else -- measured, not read off the source,
by 'test_section_2c_writes_only_the_two_diffusion_coefficients'. With 'kem = ke'
(turb_diffusion.f90:843) that is 79 of the 81 rows, and within 'ivstart:ivend' every one of them
differs from its input at every column and all four dates, so the copy-of-entry convention gives
full coverage rather than a partly vacuous comparison.

Both storages hold a LENGTH at the exit savepoint and a diffusion coefficient everywhere else,
which is why the reader offers 'stab_len_m()'/'stab_len_h()' at 'turbdiff-2c-exit' and refuses
'tkvm()'/'tkvh()' there. Section 3) multiplies them back.

FOUR OF THE SECTION'S FIVE BLOCKS ARE INACTIVE, AND NONE OF THEM IS PORTED
--------------------------------------------------------------------------
Section 2c) is 108 lines of Fortran and one of them survives the operational configuration. What
gates the other four, what the namelist says, and which test measures it:

    :1691-1700  'tfv = frm(:,ke) - ftm(:,ke)'      'rsur_sher > 0'          rsur_sher = 0.0
    :1704-1717  'tket_buoy/fshr/gshr'              'loutbms'                loutbms = .FALSE.
    :1720-1738  'vert_smooth' on 'frm' and 'frh'   'frcsmot > 0'            frcsmot = 0.0
    :1766-1793  'hlp', the phase-diffusion gradient 'ltmpcor .AND. lcpfluc' both .FALSE.

'exp.mch_icon-ch2_small' sets 'frcsmot = 0.0' explicitly (run/exp.mch_icon-ch2_small:623) and
leaves the other three at the compiled-in defaults of mo_turbdiff_config.f90 (:176 'rsur_sher',
":so far deactivated"; :273 'loutbms'; :276 'ltmpcor'; :277 'lcpfluc').

The port spec (5.4) asks for such gaps to be named rather than hoped away, and wave 2a's section
6) is the precedent: an inactive branch is left unported and pinned by a test that fails if the
branch ever becomes live. Two of the four are permanently out of reach and two are not, and the
difference is the one thing here that needs a decision outside this module:

  * 'ltmpcor' and 'lcpfluc' are 'FrozenSwitch(False)' in 'TurbulenceConfig', so the 'hlp' block
    is refused at configuration time and cannot be reached at all.
  * 'loutbms' gates only writes to 'tket_buoy', 'tket_fshr' and 'tket_gshr', which the ICON
    interfaces never pass -- they are not even in the savepoint's field list. 'turbulence.py'
    accepts the switch at any value for exactly that reason.
  * 'rsur_sher' is a plain tuning parameter with no frozen value. It is 0.0 in every
    configuration under 'icon/run/' and ICON's own comment calls it deactivated, but nothing in
    the port refuses a positive one.
  * 'frcsmot' is accepted over [0, 1] and is named in 'turbulence.py's module docstring as one
    of the eight switches "supported over the range those setups need" -- DWD operates at 0.2.
    So the configuration today promises a smoothing this section does not perform.

WHY 'vert_smooth' IS NOT PORTED
-------------------------------
Because no capture made from this experiment can validate it, at any 'frcsmot'. That is a
stronger statement than "the namelist switches it off", and it is measured by
'test_the_tke_forcing_smoothing_cannot_be_exercised_by_this_experiment': with
'imode_frcsmot = 2' -- the value MCH sets and the value 'TurbulenceConfig' freezes -- the call is
guarded by 'lcond = ANY(trop_mask > 0)', and 'trop_mask' is identically zero at all 8276
computed columns of this domain. Switzerland is not in the tropics. Even the call that did
happen would then scale by 'versmot = frcsmot*trop_mask = 0' and be the identity.

Porting it anyway would put an untestable 40-line stencil into the granule; leaving it out leaves
'frcsmot > 0' unimplemented while the configuration still accepts it. The second is the smaller
and the visible failure, so that is the choice, and it is reported to the orchestrator as a
'FrozenSwitch' decision on 'frcsmot' rather than made here.

One finding for whoever does port it: 'vert_smooth' (turb_utilities.f90:3098-3227) is NOT a
vertical recurrence, despite its '!$ACC LOOP SEQ'. 'sav_tend(:,j2)' carries the level above's
value from BEFORE it was smoothed, so the loop reads only the input profile and the whole
routine is an out-of-place three-point stencil,

    out(k) = (1-2s)*in(k) + s*(in(k-1)*dm(k-1) + in(k+1)*dm(k+1))/dm(k)

with '(1-s)' and the one available neighbour at each end -- a 'concat_where' over 'Koff[-1]' and
'Koff[1]', not a 'scan_operator'. This is the test of the package README's "Vertical recurrences"
section applied to it: the array read at 'k-1' is 'sav_tend', not 'cur_tend'.

THE ONE LIVE BLOCK IS A RECIPROCAL, NOT A DIVISION
--------------------------------------------------
'wert = z1/tke' is formed once and both coefficients are multiplied by it. Rewriting that as
'tkvm/tke' is a different computation and the reference data can tell them apart at a quarter of
its values; 'test_the_shared_reciprocal_is_not_a_division' measures it, and the stencil's module
docstring carries the numbers. This is the section's whole translation risk: there is no
multiply-add for a compiler flag to contract, no boundary row selected by a coefficient and
therefore no 'concat_where', and no vertical offset at all, so all three CPU backends run every
test here.
"""

from __future__ import annotations

from typing import NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_stability_lengths_from_diffusion_coefficients import (
    compute_stability_lengths_from_diffusion_coefficients,
)
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: The storage slots section 2c) writes, under their serialized names. Asserted against the
#: capture by 'test_section_2c_writes_only_the_two_diffusion_coefficients'.
SECTION_2C_OUTPUT_SLOTS = frozenset({"td_tkvm", "td_tkvh"})

#: The three TKE-source diagnostics the 'loutbms' block would write. They are not serialized here
#: because 'turbdiff' is not given them -- which is also what makes the block unreachable.
OPTIONAL_TKE_SOURCE_FIELDS = ("td_tket_buoy", "td_tket_fshr", "td_tket_gshr")


def _serialized(
    data_provider: sb.IconSerialDataProvider, savepoint: sb.IconTurbdiffSectionSavepoint, name: str
) -> np.ndarray:
    """A serialized slab as host memory, straight from the serializer.

    For the slots whose accessor refuses at one of the two boundaries: 'tfv()' stops at
    'turbdiff-2b-exit' because section 2c) may have overwritten it, which is exactly what a test
    that it did not has to read.
    """
    return np.asarray(data_provider.serializer.read(name, savepoint.savepoint))


class Section2c(NamedTuple):
    """One timestep of section 2c): the savepoints, the bounds and the two computed outputs."""

    entry: sb.IconTurbdiffEntrySavepoint
    before: sb.IconTurbdiffSectionSavepoint
    after: sb.IconTurbdiffSectionSavepoint
    #: Number of main levels; the exclusive end of the vertical domain, Fortran 'kem = ke'.
    ke: int
    #: Half-open range of columns 'turbdiff' computed; everything else is untouched memory with
    #: plausible values, so every comparison below is masked with it.
    columns: slice
    stability_length_for_momentum: gtx.Field
    stability_length_for_scalars: gtx.Field


def _run_section_2c(data_provider, date: str, backend) -> Section2c:
    """Run the one live program of section 2c) on the 'turbdiff-2b-exit' state of one timestep.

    Both outputs are allocated as copies of their entry state, so the two rows the Fortran does
    not write -- the model top and the surface half level -- are asserted to come out untouched
    instead of being excluded from the comparison.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="2b", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2c", date=date)

    stability_length_for_momentum = utils.copy_of(before.tkvm(), backend)
    stability_length_for_scalars = utils.copy_of(before.tkvh(), backend)

    # Fortran 'DO k=2,kem' over one-based half levels, with 'kem = ke', is
    # 'vertical_start=1, vertical_end=ke'.
    compute_stability_lengths_from_diffusion_coefficients.with_backend(backend)(
        diffusion_coefficient_for_momentum=before.tkvm(),
        diffusion_coefficient_for_scalars=before.tkvh(),
        turbulent_velocity_scale=before.tke(),
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(entry.ke()),
        offset_provider={},
    )
    return Section2c(
        entry=entry,
        before=before,
        after=after,
        ke=entry.ke(),
        columns=slice(before.ivstart(), before.ivend()),
        stability_length_for_momentum=stability_length_for_momentum,
        stability_length_for_scalars=stability_length_for_scalars,
    )


# ------------------------------------------------------------------- the section's output set --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_2c_writes_only_the_two_diffusion_coefficients(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Section 2c) changes 'tkvm' and 'tkvh', and nothing else.

    This is what bounds the section to a single stencil; 'utils.fields_that_changed' says why the
    output set is measured against the capture rather than read off the Fortran. Here it also
    does the work of four branch-coverage tests at once: 'tfv', 'frm', 'frh' and 'hlp' are all
    serialized at both savepoints, so their absence from the changed set is the direct evidence
    that none of the four optional blocks ran.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="2b", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2c", date=date)

    assert utils.fields_that_changed(data_provider, before, after) == SECTION_2C_OUTPUT_SLOTS


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_capture_carries_a_single_tke_time_level(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'tke(:,:,nvor)' is the whole serialized 'td_tke', which is the premise of the stencil call.

    The Fortran reads the time level 'nvor' of a three-dimensional 'tke'. The NWP interface runs
    with 'ntim == 1' and the reader squeezes the trailing axis away, so 'before.tke()' is that
    one time level and no index has to be chosen. Asserting it here means a recapture with more
    than one TKE time level fails with a statement instead of silently comparing the wrong slab.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="2b", date=date)

    assert entry.ntim() == 1
    assert before.nvor() == 1
    assert (
        np.asarray(data_provider.serializer.read("td_tke", before.savepoint)).shape[-1]
        == entry.ntim()
    )


# ----------------------------------------------------------------- the branches this data took --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_tke_forcing_smoothing_cannot_be_exercised_by_this_experiment(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'vert_smooth' is unreachable in this domain at ANY 'frcsmot', not merely switched off.

    Two statements, and the second is the one that decides not to port it. First, the smoothing
    did not run: 'frh' and 'frm' are byte-identical across the section, which is what
    'frcsmot = 0.0' (run/exp.mch_icon-ch2_small:623) predicts.

    Second, raising 'frcsmot' would not change that. At 'imode_frcsmot = 2' -- MCH's setting and
    the value 'TurbulenceConfig' freezes -- 'luse_mask' is true whenever the call is not an
    initialization ('lini' is false here), and the two 'vert_smooth' calls are then guarded by
    'lcond = ANY(trop_mask(ivstart:ivend) > 0)'. 'trop_mask' is identically zero over every
    computed column of this domain, so 'lcond' is false and neither call is made; and even if one
    were, 'versmot = frcsmot*trop_mask' would be zero and the routine the identity.

    So no capture from 'exp.mch_icon-ch2_small' can validate 'vert_smooth', which is why the
    block is not ported. If this test ever fails -- a tropical domain, or 'imode_frcsmot = 1' --
    the gap has become reachable and the block needs porting; delete the test rather than relax
    it. The module docstring has the structural notes for that port.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="2b", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2c", date=date)
    columns = slice(before.ivstart(), before.ivend())

    np.testing.assert_array_equal(
        before.thermal_forcing().asnumpy()[columns], after.thermal_forcing().asnumpy()[columns]
    )
    np.testing.assert_array_equal(
        before.mech_forcing().asnumpy()[columns], after.mech_forcing().asnumpy()[columns]
    )

    assert not entry.lini(), (
        "this is an initialization call, so 'luse_mask' is false and 'vert_smooth' would run "
        "unmasked at any positive 'frcsmot'."
    )
    assert not np.any(entry.trop_mask().asnumpy()[columns] > 0.0), (
        "'trop_mask' is positive somewhere in this domain, so 'lcond' would hold and the "
        "vertical smoothing of the TKE forcing becomes reachable."
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_three_other_optional_blocks_do_not_run_in_this_capture(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The NTC surface shear, the TKE-source diagnostics and the phase-diffusion gradient are dead.

    A branch-coverage statement rather than a property of the scheme, in the form the port spec
    (5.4) asks for. Each of the three is argued from what the capture contains and not from the
    namelist alone:

      * "rsur_sher > 0" would overwrite 'tfv' with 'frm(:,ke) - ftm(:,ke)'. 'tfv' is untouched,
        and it is exactly 1.0 everywhere -- 'turbtran's laminar reduction factor for vapour --
        which the difference of two forcings would not be.
      * 'loutbms' would fill 'tket_buoy', 'tket_fshr' and 'tket_gshr'. None of the three is in
        the savepoint's field list, because 'turbdiff' is not given them; the block cannot write
        anything. Its other observable, the 'ftm' fill of section 2a), is untouched too.
      * "ltmpcor .AND. lcpfluc" would put the vertical temperature gradient into 'hlp'. 'hlp' is
        untouched.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="2b", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2c", date=date)
    columns = slice(before.ivstart(), before.ivend())

    at_entry = frozenset(data_provider.serializer.fields_at_savepoint(before.savepoint))
    assert at_entry.isdisjoint(OPTIONAL_TKE_SOURCE_FIELDS)

    np.testing.assert_array_equal(before.tfv().asnumpy()[columns], np.float64(1.0))
    for name in ("td_tfv", "td_ftm", "td_hlp"):
        np.testing.assert_array_equal(
            _serialized(data_provider, before, name)[columns],
            _serialized(data_provider, after, name)[columns],
            err_msg=name,
        )


# ---------------------------------------------------------- agreement with the ICON reference --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_stability_length_for_momentum_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'ls_m' over the whole column, the two unwritten rows included.

    The comparison is not restricted to the 79 rows the stencil writes: the output started as a
    copy of the entry state, so a vertical domain that ran one row too far or one row short
    fails here rather than passing unnoticed.
    """
    run = _run_section_2c(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_stability_lengths_from_diffusion_coefficients",
        "ls_m [the 'tkvm' storage at 'turbdiff-2c-exit']",
        run.stability_length_for_momentum,
        run.after.stab_len_m(),
        columns=run.columns,
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_stability_length_for_scalars_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'ls_h' over the whole column; see the momentum test for why it is not masked to 79 rows."""
    run = _run_section_2c(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_stability_lengths_from_diffusion_coefficients",
        "ls_h [the 'tkvh' storage at 'turbdiff-2c-exit']",
        run.stability_length_for_scalars,
        run.after.stab_len_h(),
        columns=run.columns,
    )


# ------------------------------------------------------- the two things translation can get wrong --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_shared_reciprocal_is_not_a_division(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The reference performed 'x*(1/q)', and this data distinguishes it from 'x/q'.

    A statement about the reference, not about the port, and the reason the stencil forms the
    reciprocal once instead of dividing twice. 'x*(1/q)' rounds the reciprocal and then the
    product; 'x/q' rounds once. Both are one line of Python and only one of them is what ICON
    computed.

    Measured here rather than asserted from the Fortran, because "the compiler might have
    strength-reduced it back" is the objection that would otherwise stand: the reciprocal form
    reproduces the reference exactly at every one of the 653 804 written values, the division
    misses roughly a quarter of them by 1 ulp. If a future capture ever made the two agree, the
    stencil's choice would stop being observable and this test would say so by failing on its
    second assertion.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="2b", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2c", date=date)
    columns = slice(before.ivstart(), before.ivend())
    written = slice(1, entry.ke())

    velocity_scale = before.tke().asnumpy()[columns, written]
    for quantity, coefficient, length in (
        ("ls_m", before.tkvm(), after.stab_len_m()),
        ("ls_h", before.tkvh(), after.stab_len_h()),
    ):
        entered = coefficient.asnumpy()[columns, written]
        reference = length.asnumpy()[columns, written]
        np.testing.assert_array_equal(
            entered * (np.float64(1.0) / velocity_scale), reference, err_msg=quantity
        )
        assert not np.array_equal(entered / velocity_scale, reference), (
            f"'{quantity}' is the same under 'x/q' as under 'x*(1/q)' in this capture, so the "
            f"data no longer distinguishes the two and the stencil's choice is unvalidated."
        )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_surface_half_level_keeps_its_diffusion_coefficients(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """Half level 'ke1' is left as 'turbtran' produced it, and the data can tell.

    The vertical domain stops at 'kem = ke' because the turbulence model is not applied at the
    surface half level; section 3) would have nothing to multiply back there. That the row is
    unchanged is already covered by the gate tests, which compare the whole column against an
    output allocated as a copy of the entry state. What this test adds is that the row is not
    unchanged *by accident*: dividing it by the velocity scale would give a different number at
    some column, so a stencil that ran one row too far would be caught.

    The model top (row 0) gets no such statement, and deliberately: 'turbdiff' never writes it,
    it is exactly zero throughout, and zero times any reciprocal is zero -- this data cannot tell
    a stencil that included row 0 from one that did not.
    """
    run = _run_section_2c(data_provider, date, backend)
    surface = run.ke
    velocity_scale = run.before.tke().asnumpy()[run.columns, surface]

    for quantity, coefficient in (
        ("tkvm", run.before.tkvm()),
        ("tkvh", run.before.tkvh()),
    ):
        entered = coefficient.asnumpy()[run.columns, surface]
        assert not np.any(entered * (np.float64(1.0) / velocity_scale) == entered), (
            f"the surface row of '{quantity}' is unchanged by the division at some column, so "
            f"this data cannot tell whether the vertical domain stops at 'kem'."
        )

    assert not np.any(run.before.tkvm().asnumpy()[run.columns, 0]), (
        "the model top of 'tkvm' is no longer zero, so the note in this test's docstring about "
        "row 0 being indistinguishable needs revisiting."
    )
