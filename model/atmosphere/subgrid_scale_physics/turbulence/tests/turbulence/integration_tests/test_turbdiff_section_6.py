# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 6) of 'turbdiff': the preparations for the TKE diffusion.

The oracle is a real ICON run: 'turbdiff-5-exit' supplies the inputs and 'turbdiff-6-exit' the
expected outputs, for the four timesteps that 'exp.mch_icon-ch2_small' serializes.

WHAT THIS SECTION WRITES
------------------------
Four quantities in three storages, measured and not read off the source
('test_section_6_writes_only_the_three_storages' and
'test_section_6_writes_only_two_of_the_five_zaux_components'):

    'zaux(:,:,2)'  sav_prof   TKE = q**2/2, the profile the diffusion starts from   k = 2..ke1
    'zaux(:,:,3)'  expl_mom   explicit diffusion momentum of the TKE equation       k = 3..ke1
    'frh'          -          scaled flux density of circulation kinetic energy     k = 2..ke1
    'frm'          -          the same, interpolated onto the flux levels           k = 3..ke1

'frh' and 'frm' held the thermal and mechanical TKE forcing until section 5); this section
takes both storages over for the circulation term, which is why the savepoint reader has
separate accessors for the two roles.

THERE IS NO VERTICAL RECURRENCE IN THIS SECTION
-----------------------------------------------
It was expected to hold one -- the port spec's file:line index puts the "TKE virtual-profile
scan" inside it -- and it does not. The scan the index means,

    cur_prof(i,k) = (cur_prof(i,k-1) - sav_prof(i,k-1))*fakt + wert + sav_prof(i,k)

is at turb_diffusion.f90:2373 of icon 26d6b98cce, which is thirty lines below this section's
exit savepoint, inside section 8) ("Bestimmung des Zirkulationstermes als zusaetzliche
TKE-Flussdichte"). The spec cites it as ':2328', which is that same statement in the
uninstrumented upstream file; the section it belongs to was mis-attributed, not the line.

What section 6) does contain is three loops that read a neighbouring half level -- 'expl_mom'
reads the diffusion coefficient at k-1, 'frm' reads 'frh' at k-1, and the dead circulation-source
loop reads 'frm' at k+1 -- and in every one of them the array read is not the array written. They are ordinary 'Koff[-1]'
stencils, and Raschendorfer runs all three as '!$ACC LOOP GANG VECTOR COLLAPSE(2)', i.e. fully
parallel in k. The one '!$ACC LOOP SEQ' in the section, at :2283, has no k-offset access at all
and is dead here besides. Nothing below is a 'scan_operator', and the four 'Exact()' gates are
the evidence that nothing needed to be.

THE ONE PLACE THE FORTRAN'S ORDER LOOKED LIKE A DEPENDENCE
----------------------------------------------------------
'sav_prof' is written twice: first with the TKE diffusion coefficient 'c_diff*l*q' (:2144-2159),
which the next loop averages onto the flux levels as 'expl_mom', and then, three loops later,
with the TKE profile itself (:2180-2185). The coefficient therefore never reaches a savepoint
and cannot be compared against anything; it is an intermediate of
'compute_explicit_tke_diffusion_momentum', computed inside its field operator. Because it is,
the Fortran's ordering constraint disappears with the aliasing that caused it, which
'test_the_two_zaux_programs_do_not_constrain_each_others_order' asserts by running the two
programs both ways round.

WHAT THIS CAPTURE DOES NOT COVER
--------------------------------
Two whole branches of the section have no oracle here, and both are named by a test rather than
hoped away (port spec 5.4):

  * 'imode_tkediff == 1', diffusion formulated in 'q' instead of in TKE. Frozen at 2 in
    'TurbulenceConfig' and not ported; 'test_the_capture_diffuses_tke_and_not_q' pins the
    branch the data actually took.
  * the explicit circulation source at :2255-2303 ('upd_prof', 'shv', 'tket_nstc'), which needs
    'lcirflx' or 'loutthcrc'. Both are false here, 'lcirflx' is frozen '.FALSE.' in the config,
    and 'test_the_explicit_circulation_source_does_not_run_in_this_capture' shows the three
    storages are untouched. Not ported.

Neither output selects a boundary row by a coefficient, so no stencil here uses 'concat_where'
and all four are validated on 'embedded' as well as on the compiled backends.

THE MODEL TOP OF 'frh' IS NOT OBSERVABLE, AND IS POISONED INSTEAD
-----------------------------------------------------------------
Three of the four programs start below row 0, and for two of them the reference data says so:
running 'expl_mom' or 'frm' from one row higher changes that row in all 8276 computed columns.
For 'frh' it does not. 'tkvh(:,0)' is exactly zero, so 'rhon*tkvh*a_circ*len_scale' is exactly
zero at the model top, and the 'frh' this section inherits is exactly zero there too -- a
'vertical_start=0' in 'compute_cke_flux_density' would produce exactly the reference and
'test_section_6_leaves_the_rows_above_its_domains_alone' could not tell.
'test_the_model_top_of_the_cke_flux_density_is_not_observable' measures that, and
'test_the_model_top_row_of_the_diffusion_coefficient_is_not_read' closes it by filling
'tkvh(:,0)' with NaN, which is section 10)'s input-poison pattern applied at the other end of
the column.

The blind row a section WRITES -- the exposure section 10) found in 'tketens(:,ke1)' -- does not
occur here: every row each of the four programs writes differs between the two savepoints on all
four dates, so no output poison test is warranted.
"""

from __future__ import annotations

from typing import NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import turbulence
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_cke_flux_at_main_levels import (
    compute_cke_flux_at_main_levels,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_cke_flux_density import (
    compute_cke_flux_density,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_explicit_tke_diffusion_momentum import (
    compute_explicit_tke_diffusion_momentum,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_saved_tke_profile import (
    compute_saved_tke_profile,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: The storages section 6) writes, under their serialized names. Asserted against the capture by
#: 'test_section_6_writes_only_the_three_storages'.
SECTION_6_OUTPUT_SLOTS = frozenset({"td_frh", "td_frm", "td_zaux"})

#: Zero-based 'zaux' components this section writes: Fortran 'zaux(:,:,2)' is 'sav_prof' and
#: 'zaux(:,:,3)' is 'expl_mom'. 'raw_zaux' takes the zero-based index.
SAVED_TKE_PROFILE, EXPLICIT_DIFFUSION_MOMENTUM = 1, 2

#: Zero-based row of the model top. None of the four programs writes it, and the one test below
#: that names it is there because the reference data cannot show that of 'frh'.
MODEL_TOP = 0

#: Zero-based 'zvari' component holding what the Fortran still calls 'prss'. It entered
#: 'turbdiff' as the half-level air pressure and 'solve_turb_budgets' replaced it in section 3)
#: by the acceleration of the near-surface thermal circulations, which is what section 6) reads.
CIRCULATION_ACCELERATION = 0


def _low_limited_tke_diffusion_factor(config: turbulence.TurbulenceConfig) -> float:
    """'c_diff_llim' of turb_diffusion.f90:2135-2139, one scalar per run.

    A host-side quantity, not a stencil one: it depends on the namelist alone. The lower limit
    applies only while the raw circulation term is active, because that is the only path that
    divides by 'expl_mom' (section 8), and it is undone again there by the correction factor
    'c_diff/c_diff_llim'.
    """
    circulation_term_is_active = config.pat_len > 0.0 and config.ltkenst
    return max(config.epsi, config.c_diff) if circulation_term_is_active else config.c_diff


#: The factor the reference run used. The compiled-in defaults give it: 'c_diff = 0.20' and
#: 'epsi = 1e-6' (mo_turbdiff_config.f90:240, :151), and 'exp.mch_icon-ch2_small' sets only
#: 'pat_len = 750', which changes whether the limit applies but not its outcome.
#: 'test_the_c_diff_lower_limit_does_not_bind_in_this_capture' recovers it from the data.
TKE_DIFFUSION_FACTOR = _low_limited_tke_diffusion_factor(turbulence.TurbulenceConfig())


def _serialized(
    data_provider: sb.IconSerialDataProvider, savepoint: sb.IconTurbdiffSectionSavepoint, name: str
) -> np.ndarray:
    """A serialized slab as host memory, straight from the serializer.

    For the slots whose accessor refuses at one of the two boundaries: 'shv()' names the role
    section 6) would give the storage, so it raises at 'turbdiff-5-exit' -- which is exactly
    where a test that the role was never taken up has to read.
    """
    return np.asarray(data_provider.serializer.read(name, savepoint.savepoint))


class Section6(NamedTuple):
    """One timestep of section 6): the savepoints, the bounds and the four computed outputs."""

    entry: sb.IconTurbdiffEntrySavepoint
    before: sb.IconTurbdiffSectionSavepoint
    after: sb.IconTurbdiffSectionSavepoint
    ke1: int
    #: Half-open range of columns 'turbdiff' computed; everything else is untouched memory with
    #: plausible values, so every comparison below is masked with it.
    columns: slice
    saved_tke_profile: gtx.Field
    explicit_diffusion_momentum: gtx.Field
    cke_flux_density: gtx.Field
    cke_flux_at_main_levels: gtx.Field


def _run_section_6(data_provider, date: str, backend, *, poison: str | None = None) -> Section6:
    """Run all four programs of section 6) on the 'turbdiff-5-exit' state of one timestep.

    The two flux-density programs are chained -- 'compute_cke_flux_at_main_levels' consumes the
    'frh' the previous program just wrote, not the reference 'frh' -- so that a bit-exact result
    says the pair composes as the granule will run it, and not merely that each half agrees when
    handed ICON's own intermediate.

    The two 'zaux' programs are run in the opposite order to the Fortran, deliberately; see
    'test_the_two_zaux_programs_do_not_constrain_each_others_order'.

    Args:
        data_provider: The serialized archive.
        date: One of 'utils.TURBDIFF_DATES'.
        backend: The backend under test.
        poison: 'model-top-diffusion-coefficient' fills row 0 of 'tkvh' with NaN, for the one
            test that measures what this section does NOT read. It is a row
            'compute_cke_flux_density' must leave out of its vertical domain, and the reference
            data cannot say so on its own -- see
            'test_the_model_top_of_the_cke_flux_density_is_not_observable'.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="5", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="6", date=date)
    ke1 = entry.ke1()
    horizontal = dict(
        horizontal_start=gtx.int32(before.ivstart()), horizontal_end=gtx.int32(before.ivend())
    )

    scalar_diffusion_coefficient = before.tkvh()
    if poison == "model-top-diffusion-coefficient":
        values = scalar_diffusion_coefficient.asnumpy().copy()
        values[:, MODEL_TOP] = np.nan
        scalar_diffusion_coefficient = gtx.as_field(
            scalar_diffusion_coefficient.domain, values, allocator=backend
        )

    # 'zaux(:,:,2)' is 'r_cpd' and 'zaux(:,:,3)' is 'dQsat/dT' on the way in; this section takes
    # both over. The entry state is copied so that the rows it leaves alone can be asserted
    # untouched, which is what the raw accessor is for -- the named ones would refuse.
    saved_tke_profile = utils.copy_of(before.raw_zaux(SAVED_TKE_PROFILE), backend)
    explicit_diffusion_momentum = utils.copy_of(
        before.raw_zaux(EXPLICIT_DIFFUSION_MOMENTUM), backend
    )
    cke_flux_density = utils.copy_of(before.thermal_forcing(), backend)
    cke_flux_at_main_levels = utils.copy_of(before.mech_forcing(), backend)

    # Fortran 'DO k=2,ke1' over one-based half levels is 'vertical_start=1, vertical_end=ke1'.
    compute_saved_tke_profile.with_backend(backend)(
        turbulent_velocity_scale=before.tke(),
        saved_tke_profile=saved_tke_profile,
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(ke1),
        offset_provider={},
        **horizontal,
    )
    # Fortran 'DO k=3,ke1'.
    compute_explicit_tke_diffusion_momentum.with_backend(backend)(
        mixing_length=before.mixing_length(),
        turbulent_velocity_scale=before.tke(),
        air_density_at_main_levels=entry.rhoh(),
        half_level_height=entry.hhl(),
        tke_diffusion_factor=TKE_DIFFUSION_FACTOR,
        explicit_diffusion_momentum=explicit_diffusion_momentum,
        vertical_start=gtx.int32(2),
        vertical_end=gtx.int32(ke1),
        offset_provider={dims.Koff.value: dims.KDim},
        **horizontal,
    )
    compute_cke_flux_density.with_backend(backend)(
        air_density=before.rhon(),
        scalar_diffusion_coefficient=scalar_diffusion_coefficient,
        circulation_acceleration=before.effective_gradient(CIRCULATION_ACCELERATION),
        mixing_length=before.mixing_length(),
        cke_flux_density=cke_flux_density,
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(ke1),
        offset_provider={},
        **horizontal,
    )
    compute_cke_flux_at_main_levels.with_backend(backend)(
        cke_flux_density=cke_flux_density,
        mixing_length=before.mixing_length(),
        cke_flux_at_main_levels=cke_flux_at_main_levels,
        vertical_start=gtx.int32(2),
        vertical_end=gtx.int32(ke1),
        offset_provider={dims.Koff.value: dims.KDim},
        **horizontal,
    )
    return Section6(
        entry=entry,
        before=before,
        after=after,
        ke1=ke1,
        columns=slice(before.ivstart(), before.ivend()),
        saved_tke_profile=saved_tke_profile,
        explicit_diffusion_momentum=explicit_diffusion_momentum,
        cke_flux_density=cke_flux_density,
        cke_flux_at_main_levels=cke_flux_at_main_levels,
    )


# ------------------------------------------------------------------- the section's output set --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_6_writes_only_the_three_storages(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Section 6) changes 'frh', 'frm' and 'zaux', and nothing else.

    This is what bounds the four stencils; 'utils.fields_that_changed' says why the output set
    is measured against the capture rather than read off the Fortran.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="5", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="6", date=date)

    assert utils.fields_that_changed(data_provider, before, after) == SECTION_6_OUTPUT_SLOTS


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_6_writes_only_two_of_the_five_zaux_components(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'zaux' is five storages in one array; this section takes over exactly two of them.

    The one that matters is component 0: it is the Exner factor on the way in and 'upd_prof' on
    the way out of section 9), and section 6) would fill it with the explicit circulation
    increments if 'lcirflx' or 'loutthcrc' held. It does not change, which is half the evidence
    for 'test_the_explicit_circulation_source_does_not_run_in_this_capture'.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="5", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="6", date=date)
    columns = slice(before.ivstart(), before.ivend())

    changed = {
        component
        for component in range(5)
        if not np.array_equal(
            before.raw_zaux(component).asnumpy()[columns],
            after.raw_zaux(component).asnumpy()[columns],
        )
    }

    assert changed == {SAVED_TKE_PROFILE, EXPLICIT_DIFFUSION_MOMENTUM}


# ----------------------------------------------------------------- the branches this data took --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_capture_diffuses_tke_and_not_q(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'imode_tkediff == 2', pinned by the branch's own fingerprint rather than by a flag.

    The mode is not serialized, so it is established from what the run did. At
    'imode_tkediff == 1' the section would additionally multiply the discretisation momentum
    'dicke' by 'q' (turb_diffusion.f90:2194) and save 'q' rather than 'q**2/2'. Neither happened:
    'dicke' is byte-identical across the section and the saved profile is the TKE.

    The 'q' formulation is not ported -- 'imode_tkediff' is a FrozenSwitch at 2 -- so this test
    is what would notice a recapture that silently switched it on.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="5", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="6", date=date)
    columns = slice(before.ivstart(), before.ivend())
    levels = slice(1, data_provider.from_savepoint_turbdiff_entry(date=date).ke1())

    np.testing.assert_array_equal(
        before.disc_mom().asnumpy()[columns], after.disc_mom().asnumpy()[columns]
    )
    turbulent_velocity_scale = after.tke().asnumpy()[columns, levels]
    np.testing.assert_array_equal(
        after.sav_prof().asnumpy()[columns, levels],
        0.5 * (turbulent_velocity_scale * turbulent_velocity_scale),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_explicit_circulation_source_does_not_run_in_this_capture(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The 'lcirflx .OR. loutthcrc' block at :2255-2303 is dead here, so nothing ports it.

    It would write 'upd_prof' ('zaux(:,:,1)'), 'shv' and, under 'loutthcrc', 'tket_nstc'. The
    first two are untouched and the third is not even serialized, since 'tket_nstc' is not
    passed to 'turbdiff' -- which is also what makes 'loutthcrc' false.

    A branch-coverage statement, not a property of the scheme: with 'lcirflx' frozen '.FALSE.'
    in 'TurbulenceConfig' this branch has no oracle in any capture the port can make, and the
    port spec (5.4) asks for such gaps to be named. If this test ever fails the gap has closed
    and the block needs porting, so delete the test rather than relax it.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="5", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="6", date=date)
    columns = slice(before.ivstart(), before.ivend())

    assert "td_tket_nstc" not in data_provider.serializer.fields_at_savepoint(after.savepoint)
    np.testing.assert_array_equal(
        before.raw_zaux(0).asnumpy()[columns], after.raw_zaux(0).asnumpy()[columns]
    )
    np.testing.assert_array_equal(
        _serialized(data_provider, before, "td_shv")[columns],
        _serialized(data_provider, after, "td_shv")[columns],
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_c_diff_lower_limit_does_not_bind_in_this_capture(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The factor the reference used is 'c_diff' itself, recovered from 'expl_mom'.

    'c_diff' is a namelist parameter and is not serialized, so the port would otherwise be
    asserting its own assumption. Inverting

        expl_mom(k) = rhoh(k-1) * (K(k-1) + K(k))/2 / (hhl(k-1) - hhl(k)),  K = c * l * q

    for 'c' recovers it from the data. The recovery is arithmetic on already-rounded values, so
    it agrees with 0.20 to about an ulp rather than exactly; the assertion is only that the
    number is 'c_diff' and not 'epsi', which differ by five orders of magnitude.

    That the limit does not bind is what makes 'TKE_DIFFUSION_FACTOR' insensitive to
    'lcircterm', and it means the 'MAX(epsi, c_diff)' branch of :2135-2139 has no oracle here.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="5", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="6", date=date)
    columns = slice(before.ivstart(), before.ivend())
    below, above = slice(2, entry.ke1()), slice(1, entry.ke1() - 1)

    length_times_velocity = (
        before.mixing_length().asnumpy()[columns] * before.tke().asnumpy()[columns]
    )
    height = entry.hhl().asnumpy()[columns]
    recovered = (
        after.expl_mom().asnumpy()[columns, below]
        * (height[:, above] - height[:, below])
        / entry.rhoh().asnumpy()[columns, above]
        / 0.5
        / (length_times_velocity[:, above] + length_times_velocity[:, below])
    )

    config = turbulence.TurbulenceConfig()
    np.testing.assert_allclose(recovered, config.c_diff, rtol=1.0e-12)
    assert config.c_diff == TKE_DIFFUSION_FACTOR
    assert config.c_diff > config.epsi


# ---------------------------------------------------------- agreement with the ICON reference --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_saved_tke_profile_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_6(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_saved_tke_profile",
        "sav_prof [zaux(:,:,2)]",
        run.saved_tke_profile,
        run.after.sav_prof(),
        columns=run.columns,
        levels=slice(1, run.ke1),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_explicit_tke_diffusion_momentum_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_6(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_explicit_tke_diffusion_momentum",
        "expl_mom [zaux(:,:,3)]",
        run.explicit_diffusion_momentum,
        run.after.expl_mom(),
        columns=run.columns,
        levels=slice(2, run.ke1),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_cke_flux_density_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_6(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_cke_flux_density",
        "frh",
        run.cke_flux_density,
        run.after.cke_flux_density(),
        columns=run.columns,
        levels=slice(1, run.ke1),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_cke_flux_at_main_levels_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """Fed by the 'frh' the previous program computed, not by ICON's -- see '_run_section_6'."""
    run = _run_section_6(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_cke_flux_at_main_levels",
        "frm",
        run.cke_flux_at_main_levels,
        run.after.cke_flux_at_main_levels(),
        columns=run.columns,
        levels=slice(2, run.ke1),
    )


# --------------------------------------------------------------------- the vertical boundaries --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_6_leaves_the_rows_above_its_domains_alone(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The four vertical domains start at three different rows, and none of them starts at 0.

    Every output was allocated as a copy of its entry state, so a row the Fortran never writes
    has to come out as this section found it. The pair that starts at row 2 -- 'expl_mom' and
    'frm' -- is where a naive translation goes wrong, because their partner field in the same
    storage starts at row 1: the flux levels are one shorter than the half levels at the top.
    """
    run = _run_section_6(data_provider, date, backend)

    for quantity, computed, reference, unwritten in (
        ("sav_prof", run.saved_tke_profile, run.after.sav_prof(), slice(0, 1)),
        ("expl_mom", run.explicit_diffusion_momentum, run.after.expl_mom(), slice(0, 2)),
        ("frh", run.cke_flux_density, run.after.cke_flux_density(), slice(0, 1)),
        ("frm", run.cke_flux_at_main_levels, run.after.cke_flux_at_main_levels(), slice(0, 2)),
    ):
        np.testing.assert_array_equal(
            computed.asnumpy()[run.columns, unwritten],
            reference.asnumpy()[run.columns, unwritten],
            err_msg=quantity,
        )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_model_top_of_the_cke_flux_density_is_not_observable(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'frh' at the model top is zero whether or not this section writes it -- measured.

    The copy-of-entry convention makes 'test_section_6_leaves_the_rows_above_its_domains_alone'
    able to see a program that writes a row it should not, but only when the value it would
    write differs from the entry state. For 'frh' at row 0 it does not:
    'tkvh(:,0)' is exactly zero -- the model top has no diffusion coefficient, which is section
    2c)'s finding -- so the product 'rhon*tkvh*a_circ*len_scale' is exactly zero there, and the
    'frh' this section inherits is exactly zero as well. A 'vertical_start=0' in
    'compute_cke_flux_density' would therefore be invisible on all four dates.

    That is measured here rather than argued, and closed by
    'test_the_model_top_row_of_the_diffusion_coefficient_is_not_read' below. The two other
    programs whose domain starts above row 0 do not need the same treatment: running 'expl_mom'
    and 'frm' from row 1 changes that row in every one of the 8276 computed columns, so their
    boundary is already distinguished by the data.

    Section 2c)'s zero row is contagious. Anything downstream of it that multiplies by 'tkvm' or
    'tkvh' inherits a model top the reference cannot check, and this is the first place in the
    port where it has been paid for.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="5", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="6", date=date)
    columns = slice(before.ivstart(), before.ivend())

    assert not before.tkvh().asnumpy()[columns, MODEL_TOP].any(), (
        "'tkvh' at the model top is no longer exactly zero, so the poison test below now "
        "measures something weaker than it claims; re-derive the premise."
    )
    assert not before.thermal_forcing().asnumpy()[columns, MODEL_TOP].any()
    assert not after.cke_flux_density().asnumpy()[columns, MODEL_TOP].any()


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_model_top_row_of_the_diffusion_coefficient_is_not_read(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'compute_cke_flux_density' starts below the model top, shown by poisoning what is there.

    The row cannot be checked by comparing values, for the reason
    'test_the_model_top_of_the_cke_flux_density_is_not_observable' measures: both the correct
    answer and the wrong one are exactly zero. Filling 'tkvh(:,0)' with NaN removes the
    coincidence -- a program whose vertical domain reached row 0 would multiply by the NaN and
    write it into 'frh(:,0)', and the comparison here covers the whole slab, model top included,
    so the NaN fails it. A correct vertical domain never touches the row and the result is
    unchanged, bit for bit.

    'frm' is compared as well although it cannot see the poison: it reads 'frh' only at rows
    1..ke1-1. Asserting it anyway is what would catch a future fusion of the two programs that
    widened the read.
    """
    run = _run_section_6(data_provider, date, backend, poison="model-top-diffusion-coefficient")

    utils.assert_agrees_with_icon(
        "compute_cke_flux_density",
        "frh with the model-top row of 'tkvh' poisoned",
        run.cke_flux_density,
        run.after.cke_flux_density(),
        columns=run.columns,
    )
    utils.assert_agrees_with_icon(
        "compute_cke_flux_at_main_levels",
        "frm with the model-top row of 'tkvh' poisoned",
        run.cke_flux_at_main_levels,
        run.after.cke_flux_at_main_levels(),
        columns=run.columns,
        levels=slice(2, run.ke1),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_two_zaux_programs_do_not_constrain_each_others_order(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The Fortran's ordering constraint on 'sav_prof' does not survive the translation.

    In the Fortran the TKE diffusion coefficient and the saved TKE profile share the storage
    'zaux(:,:,2)', so the loop that averages the coefficient onto the flux levels HAS to run
    before the one that overwrites it -- an aliasing constraint, not a data dependence. Here the
    coefficient is an intermediate inside 'compute_explicit_tke_diffusion_momentum', which reads
    'len_scale' and 'tke' and never the saved profile, so the two programs are independent.

    '_run_section_6' already runs them in the order the Fortran forbids; this runs them in the
    Fortran's order as well and asserts the two results are bit-identical. If a later fusion
    ever routes the coefficient through the 'sav_prof' buffer to save a recomputation, this is
    what fails.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="5", date=date)
    momentum = utils.copy_of(before.raw_zaux(EXPLICIT_DIFFUSION_MOMENTUM), backend)

    compute_explicit_tke_diffusion_momentum.with_backend(backend)(
        mixing_length=before.mixing_length(),
        turbulent_velocity_scale=before.tke(),
        air_density_at_main_levels=entry.rhoh(),
        half_level_height=entry.hhl(),
        tke_diffusion_factor=TKE_DIFFUSION_FACTOR,
        explicit_diffusion_momentum=momentum,
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        vertical_start=gtx.int32(2),
        vertical_end=gtx.int32(entry.ke1()),
        offset_provider={dims.Koff.value: dims.KDim},
    )

    run = _run_section_6(data_provider, date, backend)
    np.testing.assert_array_equal(
        momentum.asnumpy()[run.columns], run.explicit_diffusion_momentum.asnumpy()[run.columns]
    )
