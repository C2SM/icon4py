# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 4) of 'turbdiff': the effective diffusion coefficients.

The oracle is a real ICON run: 'turbdiff-3-exit' supplies the inputs, 'turbdiff-4-exit' the
expected outputs, for the four timesteps that 'exp.mch_icon-ch2_small' serializes.

Section 4) (turb_diffusion.f90:1915-2052) imposes lower limits on the two vertical diffusion
coefficients the turbulence model has just produced. Raschendorfer calls them LLDCs -- Lower
Limits of Diffusion Coefficients -- and is explicit that they are tuning standing in for
transport the 1D scheme does not describe (:1970-1977). There are three of them, applied in this
order: a constant floor from the namelist, a Richardson-number dependent modification of it, and
a stratospheric enhancement above 12.5 km.

WHAT THIS SECTION WRITES
------------------------
'tkvm' and 'tkvh', and nothing else. That is measured, not read off the source:
'test_section_4_writes_only_the_two_diffusion_coefficients' compares every serialized field
across the two savepoints and exactly those two differ.

IT DOES NOT WRITE 'tke', although the Fortran of this section contains two statements that do.
Both are dead here and it is worth saying why, because the section is easy to read as a pure
diffusion-coefficient section and then to be surprised by them:

    :1944  IF (imode_tkemini == 3)  tke = tke*SQRT(MAX(1, val2/tkvh))
    :1983  IF (ltkeadapt)           tke = tke*MAX(1, val2/tkvh)

'ltkeadapt' is '(imode_tkemini == 2)' (:967) and 'TurbulenceConfig' freezes 'imode_tkemini = 1',
which is also the compiled-in default the capture ran with. So neither fires, and the measurement
above confirms it: 'td_tke' is byte-identical across the boundary on all four dates.

Two further blocks of the section are equally dead, and are not ported:

    :1999  IF (lsrfshear)  replace 'tfm'/'tfh' by the LLDC drag- and shear-factors
    :2015  IF (l3dturb)    load 'tkhm'/'tkhh', the horizontal diffusion coefficients

'lsrfshear = (rsur_sher > 0 .OR. (imode_trancnf < 4 .AND. imode_suradap >= 1))' (:968) is false
at 'rsur_sher = 0' and the frozen 'imode_suradap = 0'; 'l3dturb' is serialized and is '.FALSE.'.
'test_the_four_conditional_blocks_of_section_4_are_off_in_this_configuration' states both.

ONE PROGRAM, NOT TWO
--------------------
The Fortran is a single 'k' loop that derives 'val1' and 'val2' from one set of shared
intermediates -- the height above ground, the two masks, the stratospheric ramp -- and applies
each to its own coefficient. Splitting it per output field would duplicate all of that, so the
port is one program writing both, as section 3)'s
'compute_diffusion_coefficients_from_stability_lengths' is. It reads no vertical neighbour, so
it is a plain field operator; the 'ACC LOOP' it comes from is 'COLLAPSE(2)' and not even
declared sequential.

WHAT THE REFERENCE DATA CANNOT SEE
----------------------------------
Three branches are never taken over this capture, each of them measured by a test below rather
than left implicit:

  * the glacier branch of the 'MERGE' at :1938 -- 0 of 8276 columns satisfy
    'gz0 < 0.01 .AND. l_pat > 0';
  * the tropical widening of the stratospheric enhancement -- 'trop_mask' and 'innertrop_mask'
    are identically zero over a Swiss LAM domain, so 'x4' and 'x4i' are exactly 1 everywhere;
  * the accumulation of both surface reduction factors into the scalar floor at :1939 --
    'tkred_sfc' is exactly 1 everywhere, so a translation that read that line as two independent
    products would agree bit for bit here as well.

The port translates all three as written; these tests record that the data does not distinguish
them, so that a capture which does fails loudly instead of quietly validating nothing.

NO ROW OF THIS SECTION IS BLIND, AND THAT IS MEASURED
------------------------------------------------------
Section 4) is made of 'MAX' floors applied in place, so it is the obvious candidate for the
exposure section 10) found: a row where the floor does not bind comes out equal to what went in,
and a comparison that starts from the entry state cannot see whether the port wrote it.

It does not happen at row granularity. 72 to 74 of the 79 rows the section writes DO contain
columns where the floor leaves the coefficient alone -- and every one of those rows still
differs in at least one column, on all four dates. There is no row of 'tkvm' or 'tkvh' that this
section could skip entirely without the comparison noticing, so no output poison test is
warranted.

The two rows it must not write are safe for different reasons. Extending the vertical domain up
to row 0 changes it in all 8276 columns, so 'test_section_4_leaves_the_model_top_and_the_surface
_half_level_alone' can see that overrun. Extending it down to the surface half level is not
merely visible but impossible: 'xri' is 'ke' rows deep, not 'ke1', so a 'vertical_end=ke1'
raises out of bounds rather than computing anything. The surface boundary of this section is
enforced by the shape of an input, not by a test.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import turbulence
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_effective_diffusion_coefficients import (
    compute_effective_diffusion_coefficients,
)
from icon4py.model.common import constants
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: The four minimum diffusion coefficients the reference run was configured with [m2/s]. The
#: first two are set by '&turbdiff_nml' of 'icon/run/exp.mch_icon-ch2_small'; the two
#: stratospheric ones are not, so they are the compiled-in defaults of
#: mo_turbdiff_config.f90:133-136. None of the four is serialized, which is why they are spelled
#: out here rather than read from a 'TurbulenceConfig' -- whose 'tkhmin' default is 0.75 and
#: would silently be the wrong floor. 'test_the_capture_runs_the_namelist_minimum_diffusion_
#: coefficients' pins them against the parts of the configuration that are shared.
TKMMIN: Final[float] = 0.75  # turbdiff_nml of the experiment
TKHMIN: Final[float] = 0.50  # turbdiff_nml of the experiment
TKMMIN_STRAT: Final[float] = 4.00  # mo_turbdiff_config.f90:136
TKHMIN_STRAT: Final[float] = 0.75  # mo_turbdiff_config.f90:135


class Section4(NamedTuple):
    """One timestep of section 4): the savepoints, the bounds and the two computed outputs."""

    entry: sb.IconTurbdiffEntrySavepoint
    before: sb.IconTurbdiffSectionSavepoint
    after: sb.IconTurbdiffSectionSavepoint
    ke: int
    ke1: int
    #: Half-open range of columns 'turbdiff' actually computed; every comparison is masked with it.
    columns: slice
    #: The half levels this section writes, Fortran 'DO k=2, ke'.
    model_levels: slice
    diffusion_coefficient_for_momentum: gtx.Field
    diffusion_coefficient_for_scalars: gtx.Field


def _run_section_4(data_provider, date: str, backend) -> Section4:
    """Run the one program of section 4) on the 'turbdiff-3-exit' state of one timestep."""
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="4", date=date)
    ke, ke1 = entry.ke(), entry.ke1()
    columns = slice(before.ivstart(), before.ivend())

    # Both coefficients start as their entry state and are raised in place by the Fortran, so
    # the rows outside 'DO k=2, ke' must come out as the section found them.
    tkvm = utils.copy_of(before.tkvm(), backend)
    tkvh = utils.copy_of(before.tkvh(), backend)

    compute_effective_diffusion_coefficients.with_backend(backend)(
        diffusion_coefficient_for_momentum=before.tkvm(),
        diffusion_coefficient_for_scalars=before.tkvh(),
        height_of_half_levels=entry.hhl(),
        surface_height=utils.surface_row(entry.hhl(), ke, backend),
        inverse_richardson_number=before.xri(),
        roughness_length_times_gravity=entry.gz0(),
        pattern_length_scale=entry.l_pat(),
        surface_reduction_for_momentum=entry.tkred_sfc(),
        surface_reduction_for_scalars=entry.tkred_sfc_h(),
        tropics_mask=entry.trop_mask(),
        inner_tropics_mask=entry.innertrop_mask(),
        minimum_coefficient_for_momentum=max(constants.MOLECULAR_DIFFUSIVITY_FOR_MOMENTUM, TKMMIN),
        minimum_coefficient_for_scalars=max(constants.MOLECULAR_DIFFUSIVITY_FOR_SCALARS, TKHMIN),
        stratospheric_minimum_for_momentum=TKMMIN_STRAT,
        stratospheric_minimum_for_scalars=TKHMIN_STRAT,
        effective_diffusion_coefficient_for_momentum=tkvm,
        effective_diffusion_coefficient_for_scalars=tkvh,
        horizontal_start=gtx.int32(columns.start),
        horizontal_end=gtx.int32(columns.stop),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(ke),
        offset_provider={},
    )
    return Section4(
        entry=entry,
        before=before,
        after=after,
        ke=ke,
        ke1=ke1,
        columns=columns,
        model_levels=slice(1, ke),
        diffusion_coefficient_for_momentum=tkvm,
        diffusion_coefficient_for_scalars=tkvh,
    )


# ------------------------------------------------------------------- what the section writes --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_4_writes_only_the_two_diffusion_coefficients(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The section's true output set, measured rather than read off the source.

    In particular 'td_tke' is NOT in it. Section 4) contains two statements that scale 'tke' by
    the amplification its own lower limits applied to 'tkvh' (:1944 and :1983), and both are
    behind 'imode_tkemini' settings this configuration does not use. Asserting the measured set
    rather than the expected one is what makes that a fact about the reference data instead of a
    reading of the Fortran.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="4", date=date)

    assert utils.fields_that_changed(data_provider, before, after) == {"td_tkvm", "td_tkvh"}


def test_the_four_conditional_blocks_of_section_4_are_off_in_this_configuration() -> None:
    """The three 'tke'/'tf[mh]' branches and the 3D-turbulence block are unreachable here.

    Each is a derived flag rather than a namelist switch, so this states the derivation:

        ltkeadapt = (imode_tkemini == 2)                                              (:967)
        lsrfshear = (rsur_sher > 0 .OR. (imode_trancnf < 4 .AND. imode_suradap >= 1)) (:968)

    with 'imode_tkemini', 'imode_suradap', 'imode_trancnf' and 'l3dturb' frozen in
    'FROZEN_SWITCHES' and 'rsur_sher' defaulting to 0. 'imode_tkemini == 3', which is the third
    'tke'-adaptation, is excluded by the same freeze. If a future 'TurbulenceConfig' unfreezes
    any of them, this fails and the four blocks have to be ported before the section is complete.
    """
    config = turbulence.TurbulenceConfig()

    ltkeadapt = config.imode_tkemini == 2
    lsrfshear = config.rsur_sher > 0.0 or (config.imode_trancnf < 4 and config.imode_suradap >= 1)
    assert not ltkeadapt, "'ltkeadapt' is on: section 4) now adapts 'tke' and this port does not"
    assert config.imode_tkemini != 3, "'imode_tkemini = 3' adapts 'tke' and this port does not"
    assert not lsrfshear, "'lsrfshear' is on: section 4) now rewrites 'tfm'/'tfh'"
    assert not config.l3dturb, "'l3dturb' is on: section 4) now writes 'tkhm'/'tkhh'"


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_capture_took_the_richardson_dependent_branch(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'imode_tkvmini == 2 .AND. iini /= 1' (:1932), the one branch of the section that is live.

    'iini' is serialized and is 0 for every call the NWP interface makes; 'imode_tkvmini' is not
    serialized and is frozen at 2 in 'TurbulenceConfig', which is also the compiled-in default
    the capture ran at. The alternative, 'imode_tkvmini == 1', is a plain constant floor with no
    'xri' modification at all -- a different formulation, refused by the config rather than
    silently mistranslated, and therefore not covered by any reference data this port can make.
    """
    assert turbulence.TurbulenceConfig().imode_tkvmini == 2
    assert data_provider.from_savepoint_turbdiff_entry(date=date).iini() != 1


def test_the_capture_runs_the_namelist_minimum_diffusion_coefficients() -> None:
    """The four floors are the experiment's, and the two stratospheric ones the defaults.

    None of the four is serialized, so this cannot be read back from the archive. What it can do
    is fail if somebody assumes 'TurbulenceConfig()' describes the capture: 'tkhmin' defaults to
    0.75 there and 'exp.mch_icon-ch2_small' sets 0.5, so the scalar floor of this section is one
    of the places where the compiled-in default is NOT what produced the reference.
    """
    config = turbulence.TurbulenceConfig()

    assert config.tkmmin == TKMMIN, "the momentum floor is no longer the compiled-in default"
    assert config.tkhmin != TKHMIN, (
        "'TurbulenceConfig.tkhmin' now equals the experiment's 0.5; the warning above is stale"
    )
    assert (config.tkmmin_strat, config.tkhmin_strat) == (TKMMIN_STRAT, TKHMIN_STRAT)
    # The stratospheric block at :1952 is entered on 'tkhmin_strat > 0 .OR. tkmmin_strat > 0'.
    assert TKHMIN_STRAT > 0.0 or TKMMIN_STRAT > 0.0


# ---------------------------------------------------------------------- agreement with ICON --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_effective_diffusion_coefficients_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_4(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_effective_diffusion_coefficients",
        "tkvm",
        run.diffusion_coefficient_for_momentum,
        run.after.tkvm(),
        columns=run.columns,
        levels=run.model_levels,
    )
    utils.assert_agrees_with_icon(
        "compute_effective_diffusion_coefficients",
        "tkvh",
        run.diffusion_coefficient_for_scalars,
        run.after.tkvh(),
        columns=run.columns,
        levels=run.model_levels,
    )


# ---------------------------------------------------------------------- the untouched levels --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_4_leaves_the_model_top_and_the_surface_half_level_alone(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'DO k=2, ke' writes neither row 0 nor 'ke1', and the port's vertical domain must not either.

    Row 'ke1' carries the surface coefficients 'turbtran' produced, possibly as a tile
    aggregation; row 0 is not written by 'turbdiff' anywhere. Both are the rows a translation
    that reads 'ke' as 'ke1' would overrun, and neither would be caught by the comparison above,
    which is masked to the rows the section does write.
    """
    run = _run_section_4(data_provider, date, backend)
    computed = {
        "tkvm": (run.diffusion_coefficient_for_momentum, run.after.tkvm()),
        "tkvh": (run.diffusion_coefficient_for_scalars, run.after.tkvh()),
    }
    for name, (got, want) in computed.items():
        for label, rows in (("model top", slice(0, 1)), ("surface", slice(run.ke, run.ke1))):
            np.testing.assert_array_equal(
                got.asnumpy()[run.columns, rows],
                want.asnumpy()[run.columns, rows],
                err_msg=f"section 4) wrote the {label} of '{name}'",
            )


# -------------------------------------------------------------------------- branch coverage --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_glacier_branch_of_the_height_factor_is_never_taken(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """No column of this capture satisfies 'gz0 < 0.01 .AND. l_pat > 0' (:1938).

    The 'MERGE' there picks a steeper, offset-free growth of the minimum-diffusion factor with
    height over glaciers -- bare ice being smooth, a very small roughness length together with a
    resolved surface pattern is what identifies one. Over this domain the conjunction is empty,
    so every point takes the '0.25 + 7.5e-3*dz' branch and the reference data says nothing about
    the other one. Recording it keeps the untested case visible; if this ever fails, the branch
    has become testable and the assertion should be replaced by a comparison, not widened.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    columns = slice(before.ivstart(), before.ivend())

    smooth = entry.gz0().asnumpy()[columns] < 0.01
    patterned = entry.l_pat().asnumpy()[columns] > 0.0
    assert smooth.any(), "no column is smooth enough at all, so the conjunction is empty for "
    "a second reason and this test no longer measures what it says"
    assert patterned.any(), "no column carries a surface pattern at all, same caveat"
    assert not (smooth & patterned).any()


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_tropical_widening_of_the_stratospheric_enhancement_is_never_active(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'trop_mask' and 'innertrop_mask' are identically zero, so 'x4' and 'x4i' are exactly 1.

    exp.mch_icon-ch2_small covers the Alps. The two masks (:1958-1959) widen the 12.5-17.5 km
    transition zone of the stratospheric enhancement towards the tropopause and weaken it there,
    and with both zero the whole tropical treatment collapses to a multiplication by one --
    including 'MIN(x4, x4i)', which therefore never distinguishes its two arguments. The port
    computes both anyway; this says that the reference cannot tell whether it computes them
    correctly.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    columns = slice(before.ivstart(), before.ivend())

    assert not entry.trop_mask().asnumpy()[columns].any()
    assert not entry.innertrop_mask().asnumpy()[columns].any()


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_two_surface_reductions_cannot_be_told_apart_by_this_capture(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """':1939' accumulates BOTH reduction factors into the scalar floor, invisibly here.

    The Fortran line is two statements:

        fakt1=tkred_sfc(i)*fakt1; fakt2=tkred_sfc_h(i)*fakt1

    so 'fakt2' is 'tkred_sfc_h * tkred_sfc * fakt1' and not 'tkred_sfc_h * fakt1'. That is a
    real trap -- read as two independent products it would give a scalar floor that is too large
    exactly near the surface, where the floor binds -- but this capture cannot see it, because
    'tkred_sfc' is exactly 1 in every column while 'tkred_sfc_h' genuinely varies. Multiplying
    by an exact 1.0 is exact, so both readings agree bit for bit over all four dates.

    The port takes the accumulating reading. This test is the record that the data does not
    back it.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    columns = slice(before.ivstart(), before.ivend())

    assert (entry.tkred_sfc().asnumpy()[columns] == 1.0).all()
    assert (entry.tkred_sfc_h().asnumpy()[columns] < 1.0).any(), (
        "'tkred_sfc_h' no longer varies either, so the whole surface reduction is untested"
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_every_lower_limit_of_the_section_binds_somewhere(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The counterpart of the three tests above: what the capture DOES exercise.

    A section made entirely of clips can be validated by data that never reaches any of them,
    and then the comparison says only that the coefficients were copied. This walks the same
    statements the stencil does, in numpy, and asserts that each clip and each 'MAX' changes the
    answer somewhere -- so the bit-exactness above is evidence about the formula and not about
    a pass-through.

    Deliberately a re-derivation rather than a call of the stencil: it is asking what the
    reference data contains, not what the port computes from it.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    ke = entry.ke()
    window = (slice(before.ivstart(), before.ivend()), slice(1, ke))
    column = (window[0], np.newaxis)

    height = entry.hhl().asnumpy()[window]
    above_ground = height - entry.hhl().asnumpy()[window[0], ke][:, np.newaxis]
    xri = before.xri().asnumpy()[window]

    height_factor = 0.25 + 7.5e-3 * above_ground
    momentum_factor = entry.tkred_sfc().asnumpy()[column] * height_factor
    scalar_factor = entry.tkred_sfc_h().asnumpy()[column] * momentum_factor
    assert (momentum_factor > 1.0).any(), "'MIN(z1, fakt1)' never clips"
    assert (scalar_factor > 1.0).any(), "'MIN(z1, fakt2)' never clips"

    scaled = np.minimum(1.0, scalar_factor) * xri
    assert (scaled < 0.01).any(), "'MAX(0.01, ...)' never clips"
    assert (np.maximum(0.01, scaled) > 2.5).any(), "'MIN(2.5, ...)' never clips"

    ramp = np.minimum(1.0, 2.0e-4 * np.maximum(0.0, height - 12500.0))
    assert (ramp > 0.0).any(), "no level reaches the 12.5 km onset of the stratospheric limit"
    assert (np.sqrt(xri) < 0.25).any(), "'MAX(0.25, SQRT(xri))' never clips"
    assert (np.maximum(0.25, np.sqrt(xri)) > 1.5).any(), "'MIN(x4*1.5, ...)' never clips"

    limit = TKHMIN_STRAT * ramp * np.minimum(1.5, np.maximum(0.25, np.sqrt(xri)))
    constant_limit = max(constants.MOLECULAR_DIFFUSIVITY_FOR_SCALARS, TKHMIN) * np.minimum(
        2.5, np.maximum(0.01, scaled)
    )
    assert (limit > constant_limit).any(), "the stratospheric enhancement never wins"
    assert (np.maximum(limit, constant_limit) > before.tkvh().asnumpy()[window]).any(), (
        "the section never raises 'tkvh' at all"
    )
