# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 2a) of 'turbdiff': the three-dimensional shear complements.

The oracle is a real ICON run: 'turbdiff-1c-exit' supplies the inputs and 'turbdiff-2a-exit' the
expected outputs, for the four timesteps that 'exp.mch_icon-ch2_small' serializes.

This is the longest section of 'turbdiff' -- 300 lines against 30 to 200 for the others -- and
the only consumer anywhere in the routine of the four 'vp' fields the dycore's diffusion
supplies ('hdef2', 'hdiv', 'dwdx', 'dwdy'). It is also the section with the most branches: five
namelist selectors decide what runs, and none of the five is serialized.

WHAT THIS SECTION WRITES
------------------------
Six storages, measured and not read off the source ('test_section_2a_writes_only_the_six_storages'):

    'frm'        total mechanical forcing of the TKE equation      [1/s2]   k = 2..kem
    'xri'        '1/Ri**(2/3)', the stability factor               [-]      k = 2..ke
    'layr'       uncorrected horizontal shear length scale         [m]      per column
    'hor_scale'  corrected horizontal shear length scale           [m]      k = 2..kem
    'tket_hshr'  TKE source by the separated horizontal shear      [m2/s3]  k = 2..kem
    'hlp'        kinetic energy conversion by the SSO wakes        [m2/s3]  k = 1..kem, MAIN

'hlp' is the odd one: it is written twice in the section -- first with the separated-shear
source, then with the SSO conversion on MAIN levels -- and only the second reaches the savepoint.
See 'compute_sso_wake_energy_production'.

WHICH BRANCH OF EACH SELECTOR THE DATA TOOK
-------------------------------------------
Every one of these is established from the reference data, by a test of its own, because none of
the switches is serialized and the port must not assert its own assumption:

    itype_sher    = 2       full 3D shear, including the vertical wind
    a_hshr        = 2.0     NOT the compiled-in 1.0; recovered exactly from 'layr'
    imode_shshear = 2       Ri-corrected length scale, trace-constrained strain
    imode_tkesso in {2, 3}  Ri-reduced SSO source -- the two are INDISTINGUISHABLE here
    ltkecon       = F       no convective circulation shear

WHAT THIS CAPTURE DOES NOT COVER (port spec 5.4)
-------------------------------------------------
  * 'itype_sher' 0 and 1. Only the full three-dimensional form has an oracle here.
  * 'imode_shshear' 0 and 1. Only 2 has an oracle.
  * 'imode_tkesso = 1', excluded by the data; and the difference between 2 and 3, which this
    capture cannot see because its mesh is coarser than the 2 km the mode-3 factor keys on.
  * The convective circulation term (:1602-1614), dead at 'ltkecon = .FALSE.'.
  * The 'ftm' save of the traditional mean shear (:1375-1394): 'lssintact', 'loutbms' and
    'rsur_sher > 0' are all false, so the intermediate this port calls 'mean_shear_forcing' has
    no direct oracle. Its proxy is 'xri', which is a strictly monotone function of it and is
    compared row by row.
  * The horizontal diffusion coefficients 'tkhm'/'tkhh' (:1489-1502), dead at 'l3dturb = .FALSE.';
    they are not even passed to 'turbdiff', so the savepoint has no slot for them.

NO RECURRENCE, NO SCAN. Seventeen '!$ACC LOOP GANG VECTOR' loops -- eleven of them live in this
configuration -- and not one 'LOOP SEQ' in the whole section. The single neighbouring-level read
is 'hlp(i,k-1)' and 'dp0(i,k-1)' in the SSO interpolation, and neither array is written by the
loop that reads it.

BIT-EXACTNESS AND THE ONE EXCEPTION. Everything in the section is exact arithmetic except 'xri',
which is 'EXP(2/3*LOG(...))'. Reconstructed with numpy against ICON's own 'xri' as the input,
every other output of the section is bit-identical to the reference on all four dates; with a
self-computed 'xri' the disagreement is at the 1e-15 level and propagates to 'hor_scale',
'tket_hshr' and 'frm'. The gates below record what each backend actually produced.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import turbulence
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_effective_horizontal_shear_length_scale import (
    compute_effective_horizontal_shear_length_scale,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_inverse_richardson_number_factor import (
    compute_inverse_richardson_number_factor,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_separated_horizontal_shear_tke_source import (
    compute_separated_horizontal_shear_tke_source,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_sso_wake_energy_production import (
    compute_sso_wake_energy_production,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_three_dimensional_shear_forcing import (
    compute_three_dimensional_shear_forcing,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_total_mechanical_forcing import (
    compute_total_mechanical_forcing,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_uncorrected_horizontal_shear_length_scale import (
    compute_uncorrected_horizontal_shear_length_scale,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: The storages section 2a) writes, under their serialized names. Asserted against the capture by
#: 'test_section_2a_writes_only_the_six_storages'.
SECTION_2A_OUTPUT_SLOTS = frozenset(
    {"td_frm", "td_hlp", "td_hor_scale", "td_layr", "td_tket_hshr", "td_xri"}
)

#: 'zvari' component indices (mo_turbdiff_config.f90:62-77), zero-based as the reader takes them.
U_M, V_M = 1, 2

#: The configuration the reference run used, in the one parameter of it that differs from the
#: compiled-in default. 'test_the_capture_used_a_horizontal_shear_factor_of_two' recovers
#: 'a_hshr' from the data rather than trusting this line; every other constant section 2a) needs
#: ('akt', and 'sm_0' through the closure) is at its default in 'exp.mch_icon-ch2_small'.
CONFIG = turbulence.TurbulenceConfig(a_hshr=2.0)
PARAMS = turbulence.TurbulenceParams(CONFIG)

#: Why an uncontracted reference matters enough to assert; see section 1b), where this canary
#: was introduced. The v02 capture was built with '-Kieee -Mnofma -gpu=nofma' on the four
#: turbulence translation units, and 'make' is timestamp-driven, so a rebuild that omits them
#: silently restores contracted objects.
_CONTRACTED_REFERENCE = (
    "the reference is FMA-contracted -- was build_serialize rebuilt without "
    "ICON_FCFLAGS='-Kieee -Mnofma -gpu=nofma'? See docs/superpowers/notes/"
    "2026-08-28-serialization-recipe.md section 11."
)


def _fma() -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """A vectorised IEEE-754 fused multiply-add, 'round(a*b + c)' with the product unrounded.

    numpy has no fma ufunc and Python grew 'math.fma' only in 3.13, so this goes to the C
    library. A copy of section 1b)'s helper of the same name; it belongs in 'utils.py' and is
    duplicated here only because that file is shared.
    """
    libm = ctypes.CDLL("libm.so.6")
    libm.fma.restype = ctypes.c_double
    libm.fma.argtypes = [ctypes.c_double] * 3
    ufunc = np.frompyfunc(libm.fma, 3, 1)
    return lambda a, b, c: ufunc(a, b, c).astype(np.float64)


def _inverse_richardson_number_factor(
    mean_shear_forcing: np.ndarray, thermal_forcing: np.ndarray
) -> np.ndarray:
    """The Fortran's ':1406' in numpy, for the branch-coverage tests that need a candidate 'xri'."""
    return np.exp(
        2.0
        / 3.0
        * np.log(np.maximum(1.0e-6, mean_shear_forcing) / np.maximum(1.0e-5, thermal_forcing))
    )


class Section2a(NamedTuple):
    """One timestep of section 2a): the savepoints, the bounds and the six computed outputs."""

    entry: sb.IconTurbdiffEntrySavepoint
    before: sb.IconTurbdiffSectionSavepoint
    after: sb.IconTurbdiffSectionSavepoint
    ke: int
    ke1: int
    #: Half-open range of columns 'turbdiff' computed; everything else is untouched memory with
    #: plausible values, so every comparison below is masked with it.
    columns: slice
    #: The Fortran's 'frm' after the two mean-flow loops and before the non-turbulent additions,
    #: i.e. its 'ftm'. No savepoint holds it; see the module docstring.
    mean_shear_forcing: gtx.Field
    inverse_richardson_number_factor: gtx.Field
    uncorrected_horizontal_shear_length_scale: gtx.Field
    effective_horizontal_shear_length_scale: gtx.Field
    separated_horizontal_shear_tke_source: gtx.Field
    sso_wake_energy_production: gtx.Field
    mechanical_forcing: gtx.Field


def _run_section_2a(
    data_provider, date: str, backend, *, richardson_from_icon: bool = False
) -> Section2a:
    """Run all seven programs of section 2a) on the 'turbdiff-1c-exit' state of one timestep.

    THE PROGRAMS ARE CHAINED, not each fed with ICON's own intermediate: 'xri' is formed from the
    mean shear this run computed, 'hor_scale' from that 'xri', and so on down to 'frm'. A
    bit-exact result therefore says the seven compose as the granule will run them, and not
    merely that each one agrees when handed the reference's inputs. It also means a rounding
    difference in 'xri' -- the section's only transcendental -- reaches every gate below it,
    which is the honest way round.

    Args:
        richardson_from_icon: Substitute the reference's own 'xri' for the computed one, leaving
            everything else chained as before. That isolates the transcendental from the rest of
            the section, which is what
            'test_the_section_is_bit_exact_when_the_transcendental_comes_from_icon' measures. It
            is a diagnostic and not how the granule runs.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="1c", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2a", date=date)
    ke, ke1 = entry.ke(), entry.ke1()
    horizontal = dict(
        horizontal_start=gtx.int32(before.ivstart()), horizontal_end=gtx.int32(before.ivend())
    )
    # Fortran 'DO k=2,kem' with 'kem = ke' over one-based half levels.
    half_levels = dict(vertical_start=gtx.int32(1), vertical_end=gtx.int32(ke))

    # 'xri', 'hor_scale' and 'layr' have no accessor at 'turbdiff-1c-exit' -- this section is
    # what gives them their meaning -- so their entry state comes from the raw storage. 'layr'
    # is one value per column and 'copy_of_raw_field' only makes (Cell, K) fields, so its output
    # is allocated from the cell field of the same shape that feeds it.
    mean_shear_forcing = utils.nan_like(before.mech_forcing(), backend)
    inverse_richardson_number_factor = utils.copy_of_raw_field(before, "td_xri", backend)
    uncorrected_horizontal_shear_length_scale = utils.nan_like(entry.l_hori(), backend)
    effective_horizontal_shear_length_scale = utils.copy_of_raw_field(
        before, "td_hor_scale", backend
    )
    separated_horizontal_shear_tke_source = utils.copy_of(before.tket_hshr(), backend)
    sso_wake_energy_production = utils.copy_of(before.hlp(), backend)
    mechanical_forcing = utils.copy_of(before.mech_forcing(), backend)

    compute_three_dimensional_shear_forcing.with_backend(backend)(
        vertical_gradient_u=before.vertical_gradient(U_M),
        vertical_gradient_v=before.vertical_gradient(V_M),
        dwdx=entry.dwdx(),
        dwdy=entry.dwdy(),
        horizontal_divergence=entry.hdiv(),
        horizontal_deformation_square=entry.hdef2(),
        min_forcing=entry.fc_min(),
        mean_shear_forcing=mean_shear_forcing,
        offset_provider={},
        **horizontal,
        **half_levels,
    )
    compute_inverse_richardson_number_factor.with_backend(backend)(
        mean_shear_forcing=mean_shear_forcing,
        thermal_forcing=before.thermal_forcing(),
        inverse_richardson_number_factor=inverse_richardson_number_factor,
        offset_provider={},
        **horizontal,
        **half_levels,
    )
    if richardson_from_icon:
        inverse_richardson_number_factor = utils.copy_of(after.xri(), backend)
    compute_uncorrected_horizontal_shear_length_scale.with_backend(backend)(
        horizontal_mesh_size=entry.l_hori(),
        horizontal_shear_length_factor=CONFIG.a_hshr,
        karman_constant=CONFIG.akt,
        uncorrected_horizontal_shear_length_scale=uncorrected_horizontal_shear_length_scale,
        offset_provider={},
        **horizontal,
    )
    compute_effective_horizontal_shear_length_scale.with_backend(backend)(
        uncorrected_horizontal_shear_length_scale=uncorrected_horizontal_shear_length_scale,
        half_level_height=entry.hhl(),
        surface_height=utils.surface_row(entry.hhl(), ke, backend),
        inverse_richardson_number_factor=inverse_richardson_number_factor,
        turbulent_velocity_scale=before.tke(),
        effective_horizontal_shear_length_scale=effective_horizontal_shear_length_scale,
        offset_provider={},
        **horizontal,
        **half_levels,
    )
    compute_separated_horizontal_shear_tke_source.with_backend(backend)(
        effective_horizontal_shear_length_scale=effective_horizontal_shear_length_scale,
        horizontal_divergence=entry.hdiv(),
        horizontal_deformation_square=entry.hdef2(),
        neutral_momentum_stability_function=PARAMS.sm_0,
        separated_horizontal_shear_tke_source=separated_horizontal_shear_tke_source,
        offset_provider={},
        **horizontal,
        **half_levels,
    )
    # Fortran 'DO k=1,kem': one MAIN level higher than everything else in the section.
    compute_sso_wake_energy_production.with_backend(backend)(
        sso_tendency_u=entry.ut_sso(),
        sso_tendency_v=entry.vt_sso(),
        wind_u=entry.u(),
        wind_v=entry.v(),
        sso_wake_energy_production=sso_wake_energy_production,
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(ke),
        offset_provider={},
        **horizontal,
    )
    compute_total_mechanical_forcing.with_backend(backend)(
        mean_shear_forcing=mean_shear_forcing,
        separated_horizontal_shear_tke_source=separated_horizontal_shear_tke_source,
        sso_wake_energy_production=sso_wake_energy_production,
        layer_pressure_thickness=entry.dp0(),
        momentum_diffusion_coefficient=before.tkvm(),
        inverse_richardson_number_factor=inverse_richardson_number_factor,
        mechanical_forcing=mechanical_forcing,
        offset_provider={dims.Koff.value: dims.KDim},
        **horizontal,
        **half_levels,
    )
    return Section2a(
        entry=entry,
        before=before,
        after=after,
        ke=ke,
        ke1=ke1,
        columns=slice(before.ivstart(), before.ivend()),
        mean_shear_forcing=mean_shear_forcing,
        inverse_richardson_number_factor=inverse_richardson_number_factor,
        uncorrected_horizontal_shear_length_scale=uncorrected_horizontal_shear_length_scale,
        effective_horizontal_shear_length_scale=effective_horizontal_shear_length_scale,
        separated_horizontal_shear_tke_source=separated_horizontal_shear_tke_source,
        sso_wake_energy_production=sso_wake_energy_production,
        mechanical_forcing=mechanical_forcing,
    )


# ------------------------------------------------------------------- the section's output set --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_2a_writes_only_the_six_storages(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The section's true output set, measured rather than read off the source.

    'turbdiff' reuses its working arrays and half of this section is behind namelist switches,
    so "which fields does this section write" is a question about the run and not only about the
    code. Every field serialized at both boundaries is compared over the computed columns.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="1c", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2a", date=date)

    assert utils.fields_that_changed(data_provider, before, after) == SECTION_2A_OUTPUT_SLOTS


# ----------------------------------------------------------------- the branches this data took --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_capture_runs_the_full_three_dimensional_shear(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'itype_sher = 2', pinned by the branch's fingerprint rather than by a flag.

    The selector is not serialized, so it is established from what the run did. 'xri' is a
    strictly monotone function of the mechanical forcing at exactly the point the three
    formulations differ (:1406 reads 'frm' after the mean-flow loops and before anything else
    is added to it), so feeding each candidate through :1406 and comparing against the
    serialized 'xri' separates them:

        itype_sher = 0   the single-column shear alone, which section 1b) already computed
        itype_sher = 1   plus 'hdef2'
        itype_sher = 2   plus the vertical-wind terms and the incompressible '3*hdiv**2'

    Only the third reproduces 'xri', and the other two are wrong by 25%, not by an ulp. This is
    what makes 'compute_three_dimensional_shear_forcing' the right translation and its two
    unported branches genuinely uncovered.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="1c", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2a", date=date)
    columns = slice(before.ivstart(), before.ivend())
    levels = slice(1, entry.ke())

    def window(field) -> np.ndarray:
        return field.asnumpy()[columns, levels]

    single_column = window(before.mech_forcing())
    deformation = window(entry.hdef2())
    shear_u = window(before.vertical_gradient(U_M)) + window(entry.dwdx())
    shear_v = window(before.vertical_gradient(V_M)) + window(entry.dwdy())
    divergence = window(entry.hdiv())
    floor = entry.fc_min().asnumpy()[columns, np.newaxis]
    three_dimensional = (
        np.maximum(shear_u * shear_u + shear_v * shear_v + 3.0 * (divergence * divergence), floor)
        + deformation
    )
    thermal = window(before.thermal_forcing())
    reference = window(after.xri())

    def worst_relative_error(candidate: np.ndarray) -> float:
        return float(
            np.max(
                np.abs(_inverse_richardson_number_factor(candidate, thermal) - reference)
                / np.abs(reference)
            )
        )

    assert worst_relative_error(three_dimensional) < 1.0e-12
    assert worst_relative_error(single_column) > 1.0e-3
    assert worst_relative_error(single_column + deformation) > 1.0e-3


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_capture_used_a_horizontal_shear_factor_of_two(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'a_hshr = 2.0', not the compiled-in 1.0, recovered exactly from 'layr'.

    'layr(i) = a_hshr*akt/2 * l_hori(i)' is the one place a namelist parameter of this section
    appears alone and undivided, so inverting it recovers 'a_hshr' to the last bit rather than
    to a tolerance. Every other constant the section uses is at its 'TurbulenceConfig' default;
    'a_hshr' is not, and a port that assumed the default would be wrong by a factor of two in
    every horizontal-shear quantity below.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2a", date=date)
    columns = slice(after.ivstart(), after.ivend())

    recovered = after.layr().asnumpy()[columns] / (
        CONFIG.akt * 0.5 * entry.l_hori().asnumpy()[columns]
    )

    np.testing.assert_array_equal(recovered, np.full_like(recovered, 2.0))
    assert CONFIG.a_hshr == 2.0
    assert turbulence.TurbulenceConfig().a_hshr == 1.0


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_capture_corrects_the_shear_length_scale_by_the_richardson_number(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'imode_shshear = 2', pinned by 'hor_scale' differing from 'layr'.

    At any other value the Fortran takes 'hor_scale(i,k) = layr(i)' unchanged (:1451-1463), so
    the two arrays would be equal on every written row. They are not equal anywhere near
    everywhere: the correction spans the clipped range [0.01, 5] and the division by the
    turbulent velocity on top of it.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2a", date=date)
    columns = slice(after.ivstart(), after.ivend())
    levels = slice(1, entry.ke())

    corrected = after.hor_scale().asnumpy()[columns, levels]
    uncorrected = after.layr().asnumpy()[columns, np.newaxis]

    assert not np.any(corrected == uncorrected)
    assert CONFIG.imode_shshear == 2


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_capture_uses_the_trace_constrained_horizontal_strain(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The strain velocity is :1483's form and not :1472's, pinned by 'tket_hshr'.

    'imode_shshear = 0' would take 'hor_scale*SQRT(hdef2 + hdiv**2)', "not equal to trace of
    2D-strain tensor" in Raschendorfer's own words; every other value takes
    'hor_scale*(SQRT((fakt*hdiv)**2 + hdef2) - fakt*hdiv)', which is. Given the reference's own
    'hor_scale' the second reproduces 'tket_hshr' bit for bit and the first is out by orders of
    magnitude, so the branch is not a matter of interpretation.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2a", date=date)
    columns = slice(after.ivstart(), after.ivend())
    levels = slice(1, entry.ke())

    length_scale = after.hor_scale().asnumpy()[columns, levels]
    divergence = entry.hdiv().asnumpy()[columns, levels]
    deformation = entry.hdef2().asnumpy()[columns, levels]
    reference = after.tket_hshr().asnumpy()[columns, levels]

    def source(strain: np.ndarray) -> np.ndarray:
        return strain * strain * strain / length_scale

    # 'fakt = z1/(z2*sm_0)**2' is a reciprocal formed once and then multiplied in; dividing
    # by '(2*sm_0)**2' instead rounds twice and is off by an ulp, which this comparison sees.
    scaled = (1.0 / ((2.0 * PARAMS.sm_0) * (2.0 * PARAMS.sm_0))) * divergence
    trace_constrained = length_scale * (np.sqrt(scaled * scaled + deformation) - scaled)
    incompressible = length_scale * np.sqrt(deformation + divergence * divergence)

    assert np.array_equal(source(trace_constrained), reference)
    assert not np.array_equal(source(incompressible), reference)


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_capture_reduces_the_sso_source_by_the_richardson_number(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'imode_tkesso' is 2 or 3 and not 1, pinned by 'frm'.

    Mode 1 adds 'wert/tkvm' unreduced (:1587); modes 2 and 3 multiply it by
    'MIN(1, MAX(0.01, xri))'. Rebuilding 'frm' from the reference's own intermediates both ways
    round, the reduced form is bit-exact and the unreduced one is not close.
    """
    run_inputs = _sso_reconstruction(data_provider, date)

    assert np.array_equal(run_inputs.reduced, run_inputs.reference)
    assert not np.array_equal(run_inputs.unreduced, run_inputs.reference)


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_two_richardson_reduced_sso_modes_are_indistinguishable_in_this_capture(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """A branch-coverage statement, not a property of the scheme.

    'imode_tkesso = 3' differs from 2 by one further factor, 'MIN(1, l_hori/2000)' (:1591), an
    additional reduction on meshes finer than 2 km. 'exp.mch_icon-ch2_small' runs at
    'l_hori = 9863.8 m' on every column, so the factor is exactly 1 and NO COMPARISON AGAINST
    THIS REFERENCE DATA CAN TELL THE TWO MODES APART. What the port implements is their common
    value. If this test ever fails the capture has gained a mesh that resolves the difference,
    and mode 3 then needs its own factor and its own gate.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2a", date=date)
    columns = slice(after.ivstart(), after.ivend())

    mesh_size = entry.l_hori().asnumpy()[columns]

    np.testing.assert_array_equal(np.minimum(1.0, mesh_size / 2000.0), np.ones_like(mesh_size))


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_capture_adds_no_convective_circulation_shear(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The convective term at :1602-1614 is dead here, and 'tket_conv' is not zero.

    'ltkecon' is frozen '.FALSE.' in 'TurbulenceConfig' and is not serialized, so the evidence
    is that 'frm' is already reproduced without the term while 'tket_conv' carries real values:
    adding 'MAX(0, tket_conv/tkvm)' would change the answer. Not ported; if this test ever
    fails the branch has become live and needs porting rather than relaxing.
    """
    run_inputs = _sso_reconstruction(data_provider, date)
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="1c", date=date)
    columns = slice(before.ivstart(), before.ivend())
    levels = slice(1, entry.ke())

    convective = np.maximum(
        0.0,
        entry.tket_conv().asnumpy()[columns, levels] / before.tkvm().asnumpy()[columns, levels],
    )

    assert np.count_nonzero(convective) > 0
    assert not np.array_equal(run_inputs.reduced + convective, run_inputs.reference)
    assert not CONFIG.ltkecon


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_traditional_mean_shear_is_not_saved_in_this_capture(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'ftm' is untouched, which is why the mean-shear intermediate has no direct oracle.

    The save at :1375-1394 needs 'lssintact' ("imode_adshear == 1", frozen at 2), 'loutbms'
    (frozen '.FALSE.') or 'rsur_sher > 0' (0.0 here). None holds, so 'td_ftm' is untouched
    memory at every one of the fifteen section savepoints and the quantity this port calls
    'mean_shear_forcing' is validated only through 'xri' and 'frm'.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="1c", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2a", date=date)
    columns = slice(before.ivstart(), before.ivend())

    np.testing.assert_array_equal(before.ftm().asnumpy()[columns], after.ftm().asnumpy()[columns])
    assert CONFIG.imode_adshear == 2
    assert not CONFIG.loutbms
    assert CONFIG.rsur_sher == 0.0


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_horizontal_diffusion_coefficients_are_not_computed_in_this_capture(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The 'tkhm'/'tkhh' block at :1489-1502 is dead: 'l3dturb' is false and neither is passed.

    'turbdiff' aborts at :912 if 'l3dturb' holds without them, so their absence from the
    savepoint and 'l3dturb = .FALSE.' are the same fact seen twice. Not ported.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2a", date=date)
    serialized = data_provider.serializer.fields_at_savepoint(after.savepoint)

    assert not entry.l3dturb()
    assert "td_tkhm" not in serialized
    assert "td_tkhh" not in serialized


class _SsoReconstruction(NamedTuple):
    """'frm' rebuilt from the reference's own intermediates, with and without the Ri reduction."""

    reference: np.ndarray
    reduced: np.ndarray
    unreduced: np.ndarray


def _sso_reconstruction(data_provider: sb.IconSerialDataProvider, date: str) -> _SsoReconstruction:
    """The two 'imode_tkesso' candidates for 'frm', in numpy, from serialized quantities only."""
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="1c", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2a", date=date)
    columns = slice(before.ivstart(), before.ivend())
    ke = entry.ke()
    here, above = slice(1, ke), slice(0, ke - 1)

    def window(field, levels: slice = here) -> np.ndarray:
        return field.asnumpy()[columns, levels]

    shear_u = window(before.vertical_gradient(U_M)) + window(entry.dwdx())
    shear_v = window(before.vertical_gradient(V_M)) + window(entry.dwdy())
    divergence = window(entry.hdiv())
    floor = entry.fc_min().asnumpy()[columns, np.newaxis]
    mean_shear = np.maximum(
        shear_u * shear_u + shear_v * shear_v + 3.0 * (divergence * divergence), floor
    ) + window(entry.hdef2())

    momentum = window(before.tkvm())
    sso = after.hlp().asnumpy()[columns]
    thickness = entry.dp0().asnumpy()[columns]
    at_half_level = (sso[:, here] * thickness[:, above] + sso[:, above] * thickness[:, here]) / (
        thickness[:, here] + thickness[:, above]
    )
    source = np.maximum(0.0, -at_half_level)
    with_shear = mean_shear + window(after.tket_hshr()) / momentum
    reduction = np.minimum(1.0, np.maximum(0.01, window(after.xri())))

    return _SsoReconstruction(
        reference=window(after.mech_forcing()),
        reduced=with_shear + source / momentum * reduction,
        unreduced=with_shear + source / momentum,
    )


# ---------------------------------------------------------- agreement with the ICON reference --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_uncorrected_horizontal_shear_length_scale_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_2a(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_uncorrected_horizontal_shear_length_scale",
        "layr",
        run.uncorrected_horizontal_shear_length_scale,
        run.after.layr(),
        columns=run.columns,
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_inverse_richardson_number_factor_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_2a(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_inverse_richardson_number_factor",
        "xri",
        run.inverse_richardson_number_factor,
        run.after.xri(),
        columns=run.columns,
        levels=slice(1, run.ke),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_effective_horizontal_shear_length_scale_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_2a(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_effective_horizontal_shear_length_scale",
        "hor_scale",
        run.effective_horizontal_shear_length_scale,
        run.after.hor_scale(),
        columns=run.columns,
        levels=slice(1, run.ke),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_separated_horizontal_shear_tke_source_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_2a(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_separated_horizontal_shear_tke_source",
        "tket_hshr",
        run.separated_horizontal_shear_tke_source,
        run.after.tket_hshr(),
        columns=run.columns,
        levels=slice(1, run.ke),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_sso_wake_energy_production_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_2a(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_sso_wake_energy_production",
        "hlp [ut_sso*u + vt_sso*v]",
        run.sso_wake_energy_production,
        run.after.hlp(),
        columns=run.columns,
        levels=slice(0, run.ke),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_total_mechanical_forcing_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """Fed by the six intermediates this run computed, not by ICON's -- see '_run_section_2a'."""
    run = _run_section_2a(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_total_mechanical_forcing",
        "frm",
        run.mechanical_forcing,
        run.after.mech_forcing(),
        columns=run.columns,
        levels=slice(1, run.ke),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_section_is_bit_exact_when_the_transcendental_comes_from_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """Every other output of section 2a) is bit-identical to ICON once 'xri' is ICON's own.

    This is what makes the three tolerant gates below honest rather than a shrug. 'xri' is the
    section's only 'EXP(LOG())'; substituting the reference's value for it and leaving the rest
    of the chain untouched, 'hor_scale', 'tket_hshr' and 'frm' agree with ICON to the last bit,
    on 'embedded', 'gtfn_cpu' and 'dace_cpu' alike -- so the disagreement is one rounding of one
    libm and not an arithmetic difference anywhere in the translation.

    Ungated on purpose: 'array_equal' with no tolerance. If a future change introduces a genuine
    numerical difference in this section, this test fails while the tolerant gates would still
    pass.
    """
    run = _run_section_2a(data_provider, date, backend, richardson_from_icon=True)
    levels = slice(1, run.ke)

    for quantity, computed, reference in (
        (
            "hor_scale",
            run.effective_horizontal_shear_length_scale,
            run.after.hor_scale(),
        ),
        (
            "tket_hshr",
            run.separated_horizontal_shear_tke_source,
            run.after.tket_hshr(),
        ),
        ("frm", run.mechanical_forcing, run.after.mech_forcing()),
    ):
        np.testing.assert_array_equal(
            computed.asnumpy()[run.columns, levels],
            reference.asnumpy()[run.columns, levels],
            err_msg=quantity,
        )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_sso_wake_energy_production_is_the_fortran_expression_up_to_one_contraction(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The backend evaluated 'a*b + c*d' exactly, up to contracting at most one product.

    Section 2a)'s FMA canary, and the only expression in it of that shape. Bit for bit, with no
    tolerance: three evaluations are admissible -- no contraction, either product fused -- so
    this pins the translation without pinning the compiler, and separates "a compiler
    contracted" from "the translation is wrong" when the 'Exact()' gate fails. It also pins the
    reference to the uncontracted evaluation, which is the canary '_CONTRACTED_REFERENCE'
    describes.
    """
    run = _run_section_2a(data_provider, date, backend)
    levels = slice(0, run.ke)
    a = run.entry.ut_sso().asnumpy()[run.columns, levels]
    b = run.entry.u().asnumpy()[run.columns, levels]
    c = run.entry.vt_sso().asnumpy()[run.columns, levels]
    d = run.entry.v().asnumpy()[run.columns, levels]
    computed = run.sso_wake_energy_production.asnumpy()[run.columns, levels]

    fma = _fma()
    admissible = {
        "no contraction": a * b + c * d,
        "first product fused": fma(a, b, c * d),
        "second product fused": fma(c, d, a * b),
    }
    matches = [name for name, value in admissible.items() if np.array_equal(computed, value)]

    assert matches, "the backend did not evaluate 'a*b + c*d' by any admissible rounding"
    assert np.array_equal(
        admissible["no contraction"], run.after.hlp().asnumpy()[run.columns, levels]
    ), _CONTRACTED_REFERENCE


# --------------------------------------------------------------------- the vertical boundaries --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_2a_leaves_the_rows_outside_its_domains_alone(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The five level-dependent outputs start at two different rows and end at two others.

    Every output was allocated as a copy of its entry state, so a row the Fortran never writes
    has to come out as this section found it. The one a naive translation gets wrong is 'hlp':
    it is the only quantity here on MAIN levels, so its domain starts one row higher than
    everything else and ends one row short of the half-level surface.
    """
    run = _run_section_2a(data_provider, date, backend)

    for quantity, computed, reference, unwritten in (
        ("xri", run.inverse_richardson_number_factor, run.after.xri(), (0,)),
        (
            "hor_scale",
            run.effective_horizontal_shear_length_scale,
            run.after.hor_scale(),
            (0,),
        ),
        (
            "tket_hshr",
            run.separated_horizontal_shear_tke_source,
            run.after.tket_hshr(),
            (0, run.ke),
        ),
        ("hlp", run.sso_wake_energy_production, run.after.hlp(), (run.ke,)),
        ("frm", run.mechanical_forcing, run.after.mech_forcing(), (0, run.ke)),
    ):
        for row in unwritten:
            np.testing.assert_array_equal(
                computed.asnumpy()[run.columns, row],
                reference.asnumpy()[run.columns, row],
                err_msg=f"{quantity} row {row}",
            )
