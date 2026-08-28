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

FOUR OF THE SECTION'S FIVE BLOCKS ARE INACTIVE IN THIS CAPTURE
--------------------------------------------------------------
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
    of the switches "supported over the range those setups need" -- DWD operates at 0.2. It is
    the one of the four blocks that IS ported, and the only stencil in this package with no ICON
    reference behind it; the next section says what stands in its place.

'vert_smooth' IS PORTED AND IS THE ONE STENCIL WITHOUT AN ICON ORACLE
---------------------------------------------------------------------
UNVALIDATED AGAINST ICON DATA, and said plainly here because nothing else in this package is.
No capture made from this experiment can exercise 'vert_smooth' at any 'frcsmot' -- a stronger
statement than "the namelist switches it off", and one that is measured by
'test_the_tke_forcing_smoothing_cannot_be_exercised_by_this_experiment': with
'imode_frcsmot = 2' -- the value MCH sets and the value 'TurbulenceConfig' freezes -- the call is
guarded by 'lcond = ANY(trop_mask > 0)', and 'trop_mask' is identically zero at all 8276
computed columns of this domain. Switzerland is not in the tropics. Even a call that did happen
would scale by 'versmot = frcsmot*trop_mask = 0' and be the identity.

It is ported regardless, because 'frcsmot = 0.2' is the Fortran default and 28 configurations
under 'icon/run/' set it, the DWD global and regional operational setups among them. Refusing
'frcsmot > 0' -- which is what this package did until 'smooth_tke_forcing_vertically' landed --
locks those out, and does so over a routine that has no numerical difficulty at all: it is 24
lines of Fortran, all of it additions and one division.

What stands in place of an ICON reference, in the four tests below the gate tests:

  1. A NUMPY TRANSCRIPTION OF THE FORTRAN, '_vert_smooth_as_the_fortran_writes_it'. It is a
     statement-by-statement copy of turb_utilities.f90:3183-3225, 'sav_tend' rotation included,
     evaluated in the same order and on the SAME arrays the capture holds -- the real 'frm',
     'frh' and 'dicke' at 'turbdiff-2b-exit'. It is an independent oracle in the one way that
     matters here: it does NOT assume the stencil's reading of the loop. If 'vert_smooth' were
     a recurrence after all, the transcription would still be right and the stencil would fail.
  2. That the two readings are distinguishable at all, so the point above is not vacuous:
     'test_a_recurrence_would_give_a_different_answer' runs a third numpy version that reads
     the ALREADY SMOOTHED level above and shows it disagrees, on this data, by far more than
     rounding.
  3. Momentum-weighted conservation, 'sum_k out(k)*dm(k) == sum_k in(k)*dm(k)'. This is a
     property of the scheme rather than of the transcription, and it is what pins the two
     boundary rows: the ends carry '(1-s)' and one neighbour instead of '(1-2s)' and two,
     precisely so that every row's weights still sum to one. Writing '(1-2s)' at an end -- the
     obvious mistranslation -- breaks it.
  4. 'frcsmot = 0' is the identity, over the whole column, bit-for-bit.

What none of that can catch is a misreading of the Fortran that the transcription and the
stencil share. A first capture from a tropical or global domain should be used to gate this
stencil against ICON before it is trusted; until then it is what it says here.

WHY IT IS NOT A SCAN. 'vert_smooth' is NOT a vertical recurrence, despite its '!$ACC LOOP SEQ'.
'sav_tend(:,j2)' carries the level above's value from BEFORE it was smoothed, so the loop reads
only the input profile and the whole routine is an out-of-place three-point stencil,

    out(k) = (1-2s)*in(k) + s*(in(k-1)*dm(k-1) + in(k+1)*dm(k+1))/dm(k)

with '(1-s)' and the one available neighbour at each end -- a 'concat_where' over 'Koff[-1]' and
'Koff[1]', not a 'scan_operator'. This is the test of the package README's "Vertical recurrences"
section applied to it: the array read at 'k-1' is 'sav_tend', not 'cur_tend'.

THE ONE LIVE BLOCK IS A RECIPROCAL, NOT A DIVISION
--------------------------------------------------
'wert = z1/tke' is formed once and both coefficients are multiplied by it. Rewriting that as
'tkvm/tke' is a different computation and the reference data can tell them apart at a quarter of
its values; 'test_the_shared_reciprocal_is_not_a_division' measures it, and the stencil's module
docstring carries the numbers. That is the whole translation risk of the live block: no
multiply-add for a compiler flag to contract, no boundary row and no vertical offset, so all
three CPU backends run every one of its tests. The 'vert_smooth' tests are the exception --
'smooth_tke_forcing_vertically' selects its two boundary rows with 'concat_where', which the
embedded backend cannot execute, so the two that run it carry 'uses_concat_where' and xfail
there.
"""

from __future__ import annotations

from typing import NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_stability_lengths_from_diffusion_coefficients import (
    compute_stability_lengths_from_diffusion_coefficients,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.smooth_tke_forcing_vertically import (
    smooth_tke_forcing_vertically,
)
from icon4py.model.common import dimension as dims
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

    This is the measurement behind the module docstring's claim that
    'smooth_tke_forcing_vertically' has no ICON oracle, and it is two statements. First, the
    smoothing did not run: 'frh' and 'frm' are byte-identical across the section, which is what
    'frcsmot = 0.0' (run/exp.mch_icon-ch2_small:623) predicts.

    Second, raising 'frcsmot' would not change that. At 'imode_frcsmot = 2' -- MCH's setting and
    the value 'TurbulenceConfig' freezes -- 'luse_mask' is true whenever the call is not an
    initialization ('lini' is false here), and the two 'vert_smooth' calls are then guarded by
    'lcond = ANY(trop_mask(ivstart:ivend) > 0)'. 'trop_mask' is identically zero over every
    computed column of this domain, so 'lcond' is false and neither call is made; and even if one
    were, 'versmot = frcsmot*trop_mask' would be zero and the routine the identity.

    So no capture from 'exp.mch_icon-ch2_small' can validate the ported stencil, and the tests
    below it substitute a numpy transcription and three structural properties. If this test ever
    fails -- a tropical domain, or 'imode_frcsmot = 1' -- the section HAS become reachable and
    the stencil can and should be gated against ICON instead; that is a better position than
    this one, so delete the test rather than relax it.
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


# ------------------------------------------------- 'vert_smooth', the block with no ICON oracle --


#: The 'frcsmot' the DWD operational setups run, and the value the tests below smooth with. The
#: capture itself was made at 0.0 (run/exp.mch_icon-ch2_small:623), which is why none of this can
#: be compared against it.
OPERATIONAL_SMOOTHING_FACTOR = 0.2


def _synthetic_smoothing_mask(columns: int) -> np.ndarray:
    """A stand-in for 'trop_mask', which is identically zero in this domain.

    Every column gets its own weight, spread over the whole range the Fortran mask can take:
    0 makes 'versmot' vanish and the routine the identity, 1 makes it the full 'frcsmot', and
    everything between is what the tropics boundary produces. Per column rather than uniform,
    because 'versmot' is a per-column quantity and a uniform stand-in could not tell a stencil
    that broadcast it wrongly from one that did not.
    """
    return np.linspace(0.0, 1.0, columns)


def _vert_smooth_as_the_fortran_writes_it(
    forcing: np.ndarray, disc_mom: np.ndarray, smoothing: np.ndarray, nlev: int
) -> np.ndarray:
    """SUBROUTINE 'vert_smooth' (turb_utilities.f90:3183-3225), transcribed statement by statement.

    The oracle for a stencil that has no ICON reference, so it is deliberately a transcription of
    the Fortran ALGORITHM and not of the stencil's reading of it: the two saved columns and their
    'j0/j1/j2' rotation are carried here exactly as the Fortran carries them, and 'cur' is
    overwritten in place exactly as 'cur_tend' is. Whether that rotation amounts to a recurrence
    is then a question this function does not prejudge -- which is the whole point of it, since
    "it is not a recurrence" is the one reading the port depends on.

    'smoothing' is 'versmot(i) = vertsmot*smotfac(i)', one value per column. Arguments are
    zero-based: Fortran 'k' is column 'k - 1' here, so 'k_tp+1 = 2' is row 1 and
    'k_sf-1 = ke1-1' is row 'nlev - 1'.
    """
    current = forcing.copy()
    saved = np.zeros((current.shape[0], 2), dtype=current.dtype)
    j1, j2 = 0, 1

    remaining = 1.0 - smoothing
    k = 1  # 'k = k_tp+1'
    saved[:, j1] = current[:, k]
    current[:, k] = (
        remaining * current[:, k]
        + smoothing * current[:, k + 1] * disc_mom[:, k + 1] / disc_mom[:, k]
    )

    remaining = 1.0 - 2.0 * smoothing
    for k in range(2, nlev - 1):  # 'DO k = k_tp+2, k_sf-2'
        j1, j2 = j2, j1  # 'j0=j1; j1=j2; j2=j0'
        saved[:, j1] = current[:, k]
        current[:, k] = (
            remaining * current[:, k]
            + smoothing
            * (saved[:, j2] * disc_mom[:, k - 1] + current[:, k + 1] * disc_mom[:, k + 1])
            / disc_mom[:, k]
        )

    remaining = 1.0 - smoothing
    k = nlev - 1  # 'k = k_sf-1'
    j2 = j1
    current[:, k] = (
        remaining * current[:, k] + smoothing * saved[:, j2] * disc_mom[:, k - 1] / disc_mom[:, k]
    )
    return current


def _vert_smooth_if_it_were_a_recurrence(
    forcing: np.ndarray, disc_mom: np.ndarray, smoothing: np.ndarray, nlev: int
) -> np.ndarray:
    """The same sweep with 'sav_tend(:,j2)' replaced by the level above AFTER it was smoothed.

    The misreading the '!$ACC LOOP SEQ' invites, and the one that would make this a
    'scan_operator'. Written out so that the agreement of the port with the transcription above
    can be shown to be a statement about the Fortran rather than an artefact of two functions
    that happen to be the same.
    """
    current = forcing.copy()
    remaining = 1.0 - smoothing
    current[:, 1] = (
        remaining * current[:, 1] + smoothing * current[:, 2] * disc_mom[:, 2] / disc_mom[:, 1]
    )
    remaining = 1.0 - 2.0 * smoothing
    for k in range(2, nlev - 1):
        current[:, k] = (
            remaining * current[:, k]
            + smoothing
            * (current[:, k - 1] * disc_mom[:, k - 1] + current[:, k + 1] * disc_mom[:, k + 1])
            / disc_mom[:, k]
        )
    remaining = 1.0 - smoothing
    current[:, nlev - 1] = (
        remaining * current[:, nlev - 1]
        + smoothing * current[:, nlev - 2] * disc_mom[:, nlev - 2] / disc_mom[:, nlev - 1]
    )
    return current


class SmoothingCase(NamedTuple):
    """One forcing profile of one timestep, smoothed by the port and by the transcription.

    Every array is already masked to 'ivstart:ivend'. Outside that window the capture holds
    untouched memory, and 'dicke' is zero there -- the transcription would divide by it and fill
    the comparison with NaN, where the stencil simply never runs.
    """

    #: The profile as section 2a) left it, host memory.
    entered: np.ndarray
    #: 'dicke' as section 1a) left it, the discretisation momentum the smoothing weights with.
    disc_mom: np.ndarray
    #: 'versmot', one value per computed column.
    smoothing: np.ndarray
    #: What 'smooth_tke_forcing_vertically' produced.
    computed: np.ndarray
    #: What '_vert_smooth_as_the_fortran_writes_it' produced.
    expected: np.ndarray
    nlev: int


def _smooth(
    data_provider, date: str, backend, quantity: str, factor: float = OPERATIONAL_SMOOTHING_FACTOR
) -> SmoothingCase:
    """Run the ported smoothing and its transcription on one real forcing profile.

    The inputs are the capture's own 'frm'/'frh' and 'dicke' at 'turbdiff-2b-exit', so the
    magnitudes and the level-to-level ratios of 'disc_mom' are the ones the scheme really
    produces; only 'versmot' is synthetic, because 'trop_mask' cannot be anything but zero here.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="2b", date=date)
    nlev = entry.ke()
    columns = slice(before.ivstart(), before.ivend())

    entered = before.mech_forcing() if quantity == "frm" else before.thermal_forcing()
    disc_mom = before.disc_mom()
    forcing = entered.asnumpy()
    momentum = disc_mom.asnumpy()
    mask = _synthetic_smoothing_mask(forcing.shape[0])

    assert np.all(momentum[columns, 1:nlev] > 0.0), (
        "'dicke' is not positive on the rows the smoothing divides by, so this test would be "
        "measuring a division by zero rather than the stencil."
    )

    computed = utils.nan_like(entered, backend)
    smooth_tke_forcing_vertically.with_backend(backend)(
        tke_forcing=utils.copy_of(entered, backend),
        discretisation_momentum=utils.copy_of(disc_mom, backend),
        smoothing_mask=gtx.as_field((dims.CellDim,), mask, allocator=backend),
        smoothing_weight=factor,
        nlev=gtx.int32(nlev),
        smoothed_tke_forcing=computed,
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(nlev + 1),
        offset_provider={dims.Koff.value: dims.KDim},
    )
    entered_here = forcing[columns]
    momentum_here = momentum[columns]
    smoothing_here = factor * mask[columns]
    return SmoothingCase(
        entered=entered_here,
        disc_mom=momentum_here,
        smoothing=smoothing_here,
        computed=computed.asnumpy()[columns],
        expected=_vert_smooth_as_the_fortran_writes_it(
            entered_here, momentum_here, smoothing_here, nlev
        ),
        nlev=nlev,
    )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("quantity", ["frm", "frh"])
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_vertical_smoothing_matches_the_fortran_transcription(
    date: str, quantity: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'smooth_tke_forcing_vertically' reproduces 'vert_smooth', bit for bit.

    NOT A COMPARISON AGAINST ICON. The reference here is
    '_vert_smooth_as_the_fortran_writes_it', a numpy transcription of the Fortran; the capture
    supplies only the inputs. Everything else in this package is gated against serialized ICON
    output and this is the one thing that is not, so it is asserted with 'array_equal' and no
    tolerance -- a transcription and a translation of the same 24 lines, evaluated in the same
    order on the same doubles, have no reason to differ by even one ulp, and if they do that is
    a finding and not a tolerance.

    The comparison covers the whole column, the two rows 'vert_smooth' does not write included:
    the port copies them through because it cannot smooth in place, and a stencil whose vertical
    domain ran one row too far would show up here.
    """
    run = _smooth(data_provider, date, backend, quantity)

    assert np.array_equal(run.computed, run.expected), (
        f"'{quantity}' differs from the Fortran transcription: max abs "
        f"{np.nanmax(np.abs(run.computed - run.expected))} over "
        f"{np.count_nonzero(run.computed != run.expected)} values."
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_a_recurrence_would_give_a_different_answer(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The two readings of 'vert_smooth's loop are distinguishable on this data.

    Without this, the agreement asserted above would be worth nothing: if reading 'sav_tend' and
    reading the already-smoothed level above happened to give the same numbers, the port's claim
    that 'vert_smooth' is not a recurrence would be untested. They do not -- the difference is
    orders of magnitude above rounding -- so the transcription really is an independent statement
    about which of the two the Fortran performs.

    Pure numpy; no stencil runs here, which is why it has no 'uses_concat_where'.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="2b", date=date)
    nlev = entry.ke()
    columns = slice(before.ivstart(), before.ivend())

    mask = _synthetic_smoothing_mask(before.mech_forcing().asnumpy().shape[0])
    forcing = before.mech_forcing().asnumpy()[columns]
    disc_mom = before.disc_mom().asnumpy()[columns]
    smoothing = OPERATIONAL_SMOOTHING_FACTOR * mask[columns]

    out_of_place = _vert_smooth_as_the_fortran_writes_it(forcing, disc_mom, smoothing, nlev)
    as_a_recurrence = _vert_smooth_if_it_were_a_recurrence(forcing, disc_mom, smoothing, nlev)

    difference = np.abs(out_of_place - as_a_recurrence)
    scale = np.abs(out_of_place).max()
    assert difference.max() > 1.0e-6 * scale, (
        "reading the smoothed level above gives the same answer as reading 'sav_tend' on this "
        "data, so nothing here tests which of the two 'vert_smooth' performs."
    )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("quantity", ["frm", "frh"])
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_vertical_smoothing_conserves_the_momentum_weighted_forcing(
    date: str, quantity: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'sum_k out(k)*dm(k)' equals 'sum_k in(k)*dm(k)'; this is what pins the boundary rows.

    A property of the scheme rather than of the transcription, so it is evidence of a different
    kind than the test above: it holds for the Fortran and would hold for any correct port,
    whether or not the two agree statement by statement.

    It is the boundary rows that it constrains. Every interior row keeps '(1-2s)' of itself and
    hands 's' to each neighbour, so the weights that reach row 'k' from the whole column sum to
    one; the first and last smoothed rows have only one neighbour, and the Fortran therefore
    gives them '(1-s)' instead. Writing '(1-2s)' at an end -- the obvious mistranslation, and
    the one a spot check of the interior cannot see -- destroys the balance, which is what
    'test_a_two_sided_end_row_would_not_conserve' shows directly.

    Asserted relative to the L1 scale of the weighted profile rather than to the sum itself:
    'frh' changes sign through the column, so the sum can cancel to near zero and a relative
    tolerance on it would be vacuous.
    """
    run = _smooth(data_provider, date, backend, quantity)
    rows = slice(1, run.nlev)

    weighted_in = run.entered[:, rows] * run.disc_mom[:, rows]
    weighted_out = run.computed[:, rows] * run.disc_mom[:, rows]
    residual = np.abs(weighted_out.sum(axis=1) - weighted_in.sum(axis=1))

    assert np.all(residual <= 1.0e-12 * np.abs(weighted_in).sum(axis=1)), (
        f"'{quantity}' is not conserved by the smoothing: worst residual "
        f"{(residual / np.abs(weighted_in).sum(axis=1)).max()} relative to the L1 scale of the "
        f"weighted profile."
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_a_two_sided_end_row_would_not_conserve(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The conservation test above has teeth: '(1-2s)' at the ends breaks it.

    Pure numpy, on the same transcription with its two '1 - versmot' factors replaced by
    '1 - 2*versmot'. If this ever stopped failing, the conservation test would be passing for a
    reason that has nothing to do with the boundary rows.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="2b", date=date)
    nlev = entry.ke()
    columns = slice(before.ivstart(), before.ivend())
    rows = slice(1, nlev)

    mask = _synthetic_smoothing_mask(before.mech_forcing().asnumpy().shape[0])
    forcing = before.mech_forcing().asnumpy()[columns]
    disc_mom = before.disc_mom().asnumpy()[columns]
    smoothing = OPERATIONAL_SMOOTHING_FACTOR * mask[columns]

    wrong = _vert_smooth_as_the_fortran_writes_it(forcing, disc_mom, smoothing, nlev)
    for k in (1, nlev - 1):
        # Undo the '(1-s)*in(k)' of the end row and put '(1-2s)*in(k)' in its place.
        wrong[:, k] = wrong[:, k] - smoothing * forcing[:, k]

    weighted_in = forcing[:, rows] * disc_mom[:, rows]
    weighted_out = wrong[:, rows] * disc_mom[:, rows]
    residual = np.abs(weighted_out.sum(axis=1) - weighted_in.sum(axis=1))

    assert np.any(residual > 1.0e-12 * np.abs(weighted_in).sum(axis=1)), (
        "a two-sided end row conserves the weighted forcing on this data, so the conservation "
        "test does not constrain the boundary rows."
    )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("quantity", ["frm", "frh"])
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_vertical_smoothing_is_the_identity_at_a_vanishing_factor(
    date: str, quantity: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'frcsmot = 0' leaves the profile untouched, bit for bit and over the whole column.

    This is the configuration the reference capture actually ran, and the only statement about
    the ported stencil that the capture's own settings support. It is also what makes the guard
    in 'Turbulence.run_turbdiff' a pure optimisation rather than a branch that changes the
    answer: at 'frcsmot = 0' running the stencil and skipping it are the same computation.

    The same identity holds column by column wherever 'trop_mask' vanishes, which in this domain
    is everywhere; the synthetic mask's first column carries that case.
    """
    run = _smooth(data_provider, date, backend, quantity, factor=0.0)

    assert np.array_equal(run.computed, run.entered), (
        f"'{quantity}' was modified at 'frcsmot = 0' in "
        f"{np.count_nonzero(run.computed != run.entered)} places."
    )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_vertical_smoothing_copies_the_two_rows_the_fortran_leaves_alone(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The model top and the surface half level come out of the port unchanged.

    'vert_smooth' modifies "levels k_tp+1 until k_sf-1" and the Fortran keeps the other two by
    smoothing in place. The port cannot -- it reads both neighbours of every interior row -- so
    it copies them, and that copy has to be exact: section 3)'s circulation acceleration reads
    row 'nlev' of the thermal forcing, so a zero there would be a silent physics change rather
    than an unused row.
    """
    run = _smooth(data_provider, date, backend, "frh")

    for row, name in ((0, "the model top"), (run.nlev, "the surface half level")):
        np.testing.assert_array_equal(run.computed[:, row], run.entered[:, row], err_msg=name)
