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

WHY BIT-EXACTNESS HERE IS ONE COMPILER FLAG DEEP ON EACH SIDE
-------------------------------------------------------------
Both forcings are 'a*b + c*d', the first fused expressions of the port, and a compiler on either
side is free to contract one product into the addition -- which changes the result by one
rounding and puts bit-exactness out of reach. Neither side does, but only because both were
arranged for it: ICON's four turbulence translation units are compiled '-Kieee -Mnofma
-gpu=nofma' for the v02 capture, and 'tests/turbulence/conftest.py' sets '-ffp-contract=off' for
the compiled backends. Both stencils are gated 'Exact()' on that basis.

An agreement that rests on two build flags is not left to speak for itself.
'test_*_is_the_fortran_expression_up_to_one_contraction' asserts, bit for bit and with no
tolerance, that what the backend produced is one of the three admissible evaluations -- no
contraction, the first product fused, the second product fused -- and, in the same test, that the
reference is the uncontracted one. When the 'Exact()' gate fails, those two assertions are what
separates "somebody rebuilt without the flags" from "the translation is wrong".

NO ROW OF THIS SECTION IS BLIND, AND THAT IS MEASURED
------------------------------------------------------
A comparison that starts from the entry state cannot see a row whose entry value already equals
its exit value, which is the exposure section 10) found in 'tketens(:,ke1)' and closed with a
NaN poison. Section 1b) does not have it, on either face:

  * every row the section writes -- 'frh' rows 1..ke, 'frm' rows 1..ke-1 -- differs between
    'turbdiff-1a-exit' and 'turbdiff-1b-exit' in every one of the 8276 computed columns, on all
    four dates;
  * every row it must NOT write is distinguished too. Extending each program's vertical domain
    by one row and diffing that row changes it in all 8276 columns -- 'frh' at the model top,
    'frm' at the model top, and 'frm' at the surface half level, which is the overrun the two
    'leaves_..._alone' tests below exist to catch.

So no poison test is written here: it would assert something the data already distinguishes.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_tke_forcing_functions import (
    compute_tke_forcing_functions,
)
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


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


#: 'zvari' component indices (mo_turbdiff_config.f90:62-77), zero-based as the reader takes them.
U_M, V_M, TET_L, H2O_G = 1, 2, 3, 4


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


def _run_section_1b(data_provider, date: str, backend) -> Section1b:
    """Run section 1b) on the 'turbdiff-1a-exit' state of one timestep.

    ONE PROGRAM, TWO STATEMENTS, AND ONLY ONE VERTICAL BOUND STATED HERE. Before the merge this
    helper passed 'vertical_end=ke1' to the thermal forcing and 'vertical_end=ke' to the
    mechanical one, so the test stated both bounds independently of the stencils. It now states
    'ke1' only, and the 'kem = ke' bound lives inside 'compute_tke_forcing_functions'.

    What still constrains that bound: the reference comparison covers every written row, and
    'test_compute_mechanical_forcing_leaves_the_model_top_and_the_surface_alone' -- carried
    through the merge unchanged -- asserts row 'ke' of 'frm' against the reference, which is the
    row a one-off in the merged statement's domain would overwrite. The section docstring records
    that extending the domain by that row changes it in all 8276 computed columns, so the
    assertion bites.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="1a", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="1b", date=date)
    ke, ke1 = entry.ke(), entry.ke1()

    # 'frh' and 'frm' have no accessor at 'turbdiff-1a-exit' -- this section is what gives them
    # their meaning -- so their entry state comes from the raw storage.
    thermal_forcing = utils.copy_of_raw_field(before, "td_frh", backend)
    mechanical_forcing = utils.copy_of_raw_field(before, "td_frm", backend)

    # Fortran 'DO k=2,ke1' over one-based half levels is 'vertical_start=1, vertical_end=ke1'.
    # The shear statement inside the program stops at 'vertical_end - 1', i.e. 'kem = ke'.
    compute_tke_forcing_functions.with_backend(backend)(
        buoyancy_factor_tet_l=before.g_tet_l(),
        buoyancy_factor_h2o_g=before.g_h2o(),
        vertical_gradient_tet_l=before.vertical_gradient(TET_L),
        vertical_gradient_h2o_g=before.vertical_gradient(H2O_G),
        vertical_gradient_u=before.vertical_gradient(U_M),
        vertical_gradient_v=before.vertical_gradient(V_M),
        min_forcing=entry.fc_min(),
        thermal_forcing=thermal_forcing,
        mechanical_forcing=mechanical_forcing,
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(ke1),
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


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
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

    assert utils.fields_that_changed(data_provider, before, after) == {"td_frh", "td_frm"}


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
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
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
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
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
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
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
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
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_thermal_forcing_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_1b(data_provider, date, backend)
    levels = slice(1, run.ke1)

    utils.assert_agrees_with_icon(
        "compute_tke_forcing_functions",
        "frh",
        run.thermal_forcing,
        run.after.thermal_forcing(),
        columns=run.columns,
        levels=levels,
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_mechanical_forcing_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_1b(data_provider, date, backend)
    levels = slice(1, run.ke)

    utils.assert_agrees_with_icon(
        "compute_tke_forcing_functions",
        "frm",
        run.mechanical_forcing,
        run.after.mech_forcing(),
        columns=run.columns,
        levels=levels,
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
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
