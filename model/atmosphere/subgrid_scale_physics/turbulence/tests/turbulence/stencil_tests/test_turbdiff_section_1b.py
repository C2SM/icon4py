# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 1b) of 'turbdiff': the two basic TKE forcing functions.

The oracle is a real ICON run, not a hand-rolled fixture: 'turbdiff-1a-exit' supplies the
inputs, 'turbdiff-1b-exit' the expected outputs, for the four timesteps that
'exp.mch_icon-ch2_small' serializes.

WHAT THIS SECTION WRITES
------------------------
'frh' at half levels 2..ke1 and 'frm' at half levels 2..kem, and nothing else. That is not read
off the source but measured: 'test_section_1b_writes_only_the_two_forcings' compares every
serialized field between the two savepoints, and only those two differ. The vertical ranges
differ by one level and the difference is load-bearing -- see the two stencil docstrings.

WHY THIS SECTION IS NOT BIT-EXACT
---------------------------------
It is not bit-exact against this capture, and the mechanism is a fused multiply-add on both
sides, in different places. The Fortran statement is a sum of two products, and a compiler is
free to contract one of them into the addition:

    ICON        nvhpc, FCFLAGS '-g -O ... -acc=gpu', i.e. WITHOUT the '-Kieee -Mnofma' the port
                spec calls for, emits 'fma(c, d, a*b)' -- the second product is fused.
    gtfn_cpu    GCC at CMake Release, i.e. at GCC's default '-ffp-contract=fast' (gt4py.next
                has no equivalent of gt4py.cartesian's '-ffp-contract=off'), emits
                'fma(a, b, c*d)' -- the first product is fused.
    embedded    numpy, no contraction at all.

Three correctly rounded evaluations of one expression, differing only in which single rounding
is elided. 'test_*_is_the_fortran_expression_up_to_one_contraction' asserts exactly that, bit
for bit and with no tolerance: whatever the backend produced equals one of the three admissible
evaluations. That is a stronger statement than any 'rtol', and it is what distinguishes "the
translation is right and a compiler contracted it" from "the translation is wrong".

The consequence for the gate registry is in the module-level 'PROPOSED_GATES' below.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_mechanical_forcing import (
    compute_mechanical_forcing,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_thermal_forcing import (
    compute_thermal_forcing,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import definitions, serialbox as sb

from .. import gate_registry
from ..fixtures import *  # noqa: F403


#: The four timesteps 'exp.mch_icon-ch2_small' serializes.
#: Why an uncontracted reference matters enough to assert. The v02 capture was built with
#: ICON_FCFLAGS='-Kieee -Mnofma -gpu=nofma' on the four turbulence translation units, so ICON
#: contracts nothing and bit-exactness is reachable. 'make' is timestamp-driven: a later rebuild
#: that omits those flags recompiles them contracted, and then every 'a*b + c*d' gate in the port
#: fails at ~1 eps with nothing to say why. This assertion says why.
_CONTRACTED_REFERENCE = (
    "the reference is FMA-contracted -- was build_serialize rebuilt without "
    "ICON_FCFLAGS='-Kieee -Mnofma -gpu=nofma'? See docs/superpowers/notes/"
    "2026-08-28-serialization-recipe.md section 11."
)


TURBDIFF_DATES = (
    "2020-12-10T06:01:00.000",
    "2020-12-10T06:01:20.000",
    "2020-12-10T06:01:40.000",
    "2020-12-10T06:02:00.000",
)

#: 'zvari' component indices (mo_turbdiff_config.f90:62-77), zero-based as the reader takes them.
U_M, V_M, TET_L, H2O_G = 1, 2, 3, 4

#: What this task would put in 'tests/turbulence/gate_registry.py' if it owned that file. It does
#: not -- several section ports run in parallel and the registry is edited by one hand -- so the
#: proposal lives here and 'test_*_agrees_with_icon_within_its_gate' skips with it until the
#: entry lands, at which point that test starts asserting instead.
#:
#: Both numbers are measured on exp.mch_icon-ch2_small, all four timesteps, gtfn_cpu and
#: embedded, over 'ivstart:ivend' and the computed half levels only. The reason is NOT in
#: 'gate_registry.Reason': FMA contraction is neither re-association (the operand order is
#: unchanged) nor any of the other three members. Per docs/gates.md that makes this a finding to
#: escalate, not a tolerance to grant -- either the reference is re-captured with '-Kieee
#: -Mnofma', or the spec gains a fifth 'Reason'.
PROPOSED_GATES = {
    # frh = g_tet*d(tet_l) + g_h2o*d(h2o_g) is a difference of two same-sign-dominated terms and
    # cancels hard: max rel err 9.1e-12 sits where |frh| ~ 1e-11 against terms of order 1e-1.
    # Measured relative to the magnitude of the operands the number is ~1 ULP everywhere.
    "compute_thermal_forcing": dict(rtol=1.0e-11, measured_max_rel_err=9.2e-12),
    # frm = MAX(du**2 + dv**2, fc_min) has no cancellation, so the contraction stays at 1 ULP.
    "compute_mechanical_forcing": dict(rtol=1.0e-15, measured_max_rel_err=2.3e-16),
}

experiment_for_turbulence = pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.MCH_ICON_CH2_SMALL],
    ids=lambda d: d.name,
)


class Section1b(NamedTuple):
    """One timestep of section 1b): the two savepoints, the bounds and the computed outputs."""

    entry: sb.IconTurbdiffEntrySavepoint
    before: sb.IconTurbdiffSectionSavepoint
    after: sb.IconTurbdiffSectionSavepoint
    ke: int
    ke1: int
    #: Half-open range of columns that 'turbdiff' actually computed; everything else is untouched
    #: memory with plausible values, so every comparison below is masked with it.
    columns: slice
    thermal_forcing: gtx.Field
    mechanical_forcing: gtx.Field


def _fma() -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """A vectorised IEEE-754 fused multiply-add, 'round(a*b + c)' with the product unrounded.

    numpy has no fma ufunc and Python grew 'math.fma' only in 3.13, so this goes to the C
    library. The reference data was produced by a build that contracts, and reproducing that
    contraction bit for bit is the whole point of the tests that use this.
    """
    libm = ctypes.CDLL("libm.so.6")
    libm.fma.restype = ctypes.c_double
    libm.fma.argtypes = [ctypes.c_double] * 3
    ufunc = np.frompyfunc(libm.fma, 3, 1)
    return lambda a, b, c: ufunc(a, b, c).astype(np.float64)


def _output_field_from_entry_state(savepoint, name: str, num_cells: int, backend) -> gtx.Field:
    """Allocate an output field holding what the Fortran storage held on entry to the section.

    The convention for every section of this port: an output field starts as the *entry*
    savepoint's contents of the same storage, so the whole slab can be compared against the exit
    savepoint. Levels the section does not write must then come out unchanged, which turns a
    wrong vertical domain into a failed assertion instead of an invisible one. Zero-filling or
    NaN-filling would make those levels differ for a reason that says nothing about the stencil.
    """
    buffer = np.asarray(savepoint.raw_field(name))[:num_cells]
    return gtx.as_field((dims.CellDim, dims.KDim), np.ascontiguousarray(buffer), allocator=backend)


def _run_section_1b(data_provider, date: str, backend) -> Section1b:
    """Run both stencils of section 1b) on the 'turbdiff-1a-exit' state of one timestep."""
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="1a", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="1b", date=date)
    ke, ke1 = entry.ke(), entry.ke1()
    num_cells = before.g_tet_l().shape[0]

    thermal_forcing = _output_field_from_entry_state(before, "td_frh", num_cells, backend)
    mechanical_forcing = _output_field_from_entry_state(before, "td_frm", num_cells, backend)

    # Fortran 'DO k=2,ke1' over one-based half levels is 'vertical_start=1, vertical_end=ke1'.
    compute_thermal_forcing.with_backend(backend)(
        buoyancy_factor_tet_l=before.g_tet_l(),
        buoyancy_factor_h2o_g=before.g_h2o(),
        vertical_gradient_tet_l=before.vertical_gradient(TET_L),
        vertical_gradient_h2o_g=before.vertical_gradient(H2O_G),
        thermal_forcing=thermal_forcing,
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(ke1),
        offset_provider={},
    )
    # Fortran 'DO k=2,kem' with 'kem = ke' stops one half level higher.
    compute_mechanical_forcing.with_backend(backend)(
        vertical_gradient_u=before.vertical_gradient(U_M),
        vertical_gradient_v=before.vertical_gradient(V_M),
        min_forcing=entry.fc_min(),
        mechanical_forcing=mechanical_forcing,
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(ke),
        offset_provider={},
    )
    return Section1b(
        entry=entry,
        before=before,
        after=after,
        ke=ke,
        ke1=ke1,
        columns=slice(before.ivstart(), before.ivend()),
        thermal_forcing=thermal_forcing,
        mechanical_forcing=mechanical_forcing,
    )


def _assert_within_gate(stencil: str, computed: np.ndarray, expected: np.ndarray) -> None:
    """Compare against the ICON reference under the gate declared for 'stencil'.

    Skips while the stencil has no registry entry rather than defaulting to a tolerance: a
    silent default is the drift 'gate_registry' exists to prevent, and inventing an entry here
    would put the threshold in the one place review does not look for it.

    The skip is temporary scaffolding, not the intended behaviour. Once the entries below land
    in the registry, delete the 'except' branch so that a missing gate is a failure -- which is
    what the section 1a) tests already do.
    """
    try:
        gate = gate_registry.gate_for(stencil)
    except gate_registry.UnregisteredStencilError:
        proposal = PROPOSED_GATES[stencil]
        pytest.skip(
            f"'{stencil}' has no entry in 'gate_registry.GATES' yet; the measured proposal is "
            f"{proposal}, and see this module's docstring for why 'Reason' has no member that "
            "fits."
        )

    if isinstance(gate, gate_registry.Exact):
        np.testing.assert_array_equal(computed, expected)
    else:
        np.testing.assert_allclose(computed, expected, rtol=gate.rtol, atol=0.0)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_section_1b_writes_only_the_two_forcings(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The section's true output set, measured rather than read off the source.

    'turbdiff' reuses its working arrays, so "which fields does this section write" is a
    question about the run and not only about the code. Every field serialized at both
    boundaries is compared over the computed columns; exactly 'td_frh' and 'td_frm' differ.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="1a", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="1b", date=date)
    columns = slice(before.ivstart(), before.ivend())

    def masked(savepoint, name: str) -> np.ndarray:
        buffer = np.asarray(data_provider.serializer.read(name, savepoint.savepoint))
        return buffer[columns] if buffer.ndim >= 2 else buffer

    names = sorted(data_provider.serializer.fields_at_savepoint(before.savepoint))
    differing = [
        name
        for name in names
        if not np.array_equal(masked(before, name), masked(after, name), equal_nan=True)
    ]

    assert differing == ["td_frh", "td_frm"]


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_compute_thermal_forcing_leaves_the_model_top_alone(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """Half level 1 of 'frh' is never written, so it must come out as the section found it."""
    run = _run_section_1b(data_provider, date, backend)

    np.testing.assert_array_equal(
        run.thermal_forcing.asnumpy()[run.columns, 0],
        run.after.thermal_forcing().asnumpy()[run.columns, 0],
    )


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_compute_mechanical_forcing_leaves_the_model_top_and_the_surface_alone(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'frm' runs to 'kem = ke' only.

    Half level 1 is never written and half level 'ke1' still holds what 'turbtran' put there, so
    writing either would be a vertical-domain error. Both are asserted because 'ke1' is the one
    a naive translation gets wrong: the thermal forcing above does run to 'ke1'.
    """
    run = _run_section_1b(data_provider, date, backend)
    computed = run.mechanical_forcing.asnumpy()
    expected = run.after.mech_forcing().asnumpy()

    np.testing.assert_array_equal(computed[run.columns, 0], expected[run.columns, 0])
    np.testing.assert_array_equal(computed[run.columns, run.ke], expected[run.columns, run.ke])


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_compute_thermal_forcing_is_the_fortran_expression_up_to_one_contraction(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The backend evaluated 'a*b + c*d' exactly, up to contracting at most one product.

    Bit for bit, with no tolerance. Three evaluations are admissible -- no contraction, the
    first product fused, the second product fused -- so this pins the translation without
    pinning the compiler, and separates 'a compiler contracted' from 'the translation is wrong'
    when the Exact() gate fails.

    It also pins the reference to the uncontracted evaluation, which is the canary described on
    '_CONTRACTED_REFERENCE'.
    """
    run = _run_section_1b(data_provider, date, backend)
    levels = slice(1, run.ke1)
    a = run.before.g_tet_l().asnumpy()[run.columns, levels]
    b = run.before.vertical_gradient(TET_L).asnumpy()[run.columns, levels]
    c = run.before.g_h2o().asnumpy()[run.columns, levels]
    d = run.before.vertical_gradient(H2O_G).asnumpy()[run.columns, levels]
    computed = run.thermal_forcing.asnumpy()[run.columns, levels]

    fma = _fma()
    admissible = {
        "no contraction": a * b + c * d,
        "first product fused": fma(a, b, c * d),
        "second product fused": fma(c, d, a * b),
    }
    matches = [name for name, value in admissible.items() if np.array_equal(computed, value)]

    assert matches, "the backend did not evaluate 'a*b + c*d' by any admissible rounding"
    assert np.array_equal(
        admissible["no contraction"],
        run.after.thermal_forcing().asnumpy()[run.columns, levels],
    ), _CONTRACTED_REFERENCE


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_compute_mechanical_forcing_is_the_fortran_expression_up_to_one_contraction(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """As for the thermal forcing: 'MAX(u**2 + v**2, fc_min)' up to one contracted product."""
    run = _run_section_1b(data_provider, date, backend)
    levels = slice(1, run.ke)
    u = run.before.vertical_gradient(U_M).asnumpy()[run.columns, levels]
    v = run.before.vertical_gradient(V_M).asnumpy()[run.columns, levels]
    floor = run.entry.fc_min().asnumpy()[run.columns, np.newaxis]
    computed = run.mechanical_forcing.asnumpy()[run.columns, levels]

    fma = _fma()
    admissible = {
        "no contraction": u * u + v * v,
        "u**2 fused": fma(u, u, v * v),
        "v**2 fused": fma(v, v, u * u),
    }
    matches = [
        name
        for name, value in admissible.items()
        if np.array_equal(computed, np.maximum(value, floor))
    ]

    assert matches, "the backend did not evaluate 'u**2 + v**2' by any admissible rounding"
    assert np.array_equal(
        np.maximum(admissible["no contraction"], floor),
        run.after.mech_forcing().asnumpy()[run.columns, levels],
    ), _CONTRACTED_REFERENCE


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_compute_thermal_forcing_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_1b(data_provider, date, backend)
    levels = slice(1, run.ke1)

    _assert_within_gate(
        "compute_thermal_forcing",
        run.thermal_forcing.asnumpy()[run.columns, levels],
        run.after.thermal_forcing().asnumpy()[run.columns, levels],
    )


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_compute_mechanical_forcing_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_1b(data_provider, date, backend)
    levels = slice(1, run.ke)

    _assert_within_gate(
        "compute_mechanical_forcing",
        run.mechanical_forcing.asnumpy()[run.columns, levels],
        run.after.mech_forcing().asnumpy()[run.columns, levels],
    )


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_the_fc_min_floor_is_never_reached_in_this_capture(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """A branch-coverage canary, not a property of the scheme.

    'MAX(du**2 + dv**2, fc_min)' has an oracle for one of its two branches only. Here
    'fc_min = (0.01/9863.78)**2 = 1.03e-12' against a smallest resolved shear of 1.4e-11, a
    factor of thirteen clear, so the floor never binds anywhere in the four timesteps and no
    comparison against this reference data can detect a mistranslated floor. Port spec section
    5.4 asks for such gaps to be named rather than hoped away; if this test ever fails, the gap
    has closed and the assertion should be deleted rather than widened.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="1a", date=date)
    columns = slice(before.ivstart(), before.ivend())

    shear_squared = (
        before.vertical_gradient(U_M).asnumpy()[columns, 1 : entry.ke()] ** 2
        + before.vertical_gradient(V_M).asnumpy()[columns, 1 : entry.ke()] ** 2
    )
    floor = entry.fc_min().asnumpy()[columns, np.newaxis]

    assert (shear_squared > floor).all()
