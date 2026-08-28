# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 1a) of 'turbdiff', the vertical gradients.

The oracle is serialized ICON, not a hand-written reference: each stencil is run on the fields
of 'turbdiff-0-exit' and compared against 'turbdiff-1a-exit', for all four timesteps of
exp.mch_icon-ch2_small. A numpy re-implementation would only restate the translation and would
agree with it for the same reason it agrees with itself.

THE OUTPUT SET IS ESTABLISHED, NOT ASSUMED. Section 1a) writes exactly four storage slots --
'lays', 'hlp', 'dicke' and 'zvari' -- and 'test_section_1a_writes_exactly_four_slots' asserts
that by comparing every serialized field across the two savepoints. That is what fixes the
scope of the three stencils below; anything else that changed would mean the section does more
than they do.

EVERY COMPARISON IS MASKED TO 'ivstart:ivend'. The hook writes the whole 'nproma' slab but the
scheme only loops over that window, and what lies outside is untouched memory holding plausible
values rather than NaN, so an unmasked comparison fails looking exactly like a physics bug.

Each output field is allocated as a COPY OF ITS ENTRY STATE and compared over the whole column,
so the rows the Fortran leaves alone -- the model top of 'hlp', 'dicke' and the five gradients,
and the surface row of 'hlp' and 'dicke' -- are asserted to be untouched rather than ignored.
'lays' is undefined before this section, so it is allocated NaN-filled instead, which makes an
unwritten column inside the window a failure rather than a coincidence.

'compute_vertical_gradients_of_conserved_variables' selects its reciprocal depth per row with
'concat_where', which the embedded backend cannot execute (see 'model/testing/filters.py'), so its
two tests carry 'uses_concat_where' and xfail there. The compiled backends still assert the whole
gradient profile bit-exactly, and the second of the two tests pins the row selection itself by
poisoning the row the surface branch must not read. Why the surface row is not a program of its
own is written down in the package README, section "Boundary rows".
"""

from __future__ import annotations

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_inverse_layer_depth_and_tke_discretisation_momentum import (
    compute_inverse_layer_depth_and_tke_discretisation_momentum,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_surface_transfer_ratios import (
    compute_surface_transfer_ratios,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_vertical_gradients_of_conserved_variables import (
    compute_vertical_gradients_of_conserved_variables,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import definitions, serialbox as sb

from .. import gate_registry
from ..fixtures import *  # noqa: F403


#: The four timesteps 'exp.mch_icon-ch2_small' serializes.
TURBDIFF_DATES = (
    "2020-12-10T06:01:00.000",
    "2020-12-10T06:01:20.000",
    "2020-12-10T06:01:40.000",
    "2020-12-10T06:02:00.000",
)

#: The storage slots section 1a) writes, as the four Fortran arrays they are serialized under.
#: Asserted against the data by 'test_section_1a_writes_exactly_four_slots'.
SECTION_1A_OUTPUT_SLOTS = frozenset({"td_dicke", "td_hlp", "td_lays", "td_zvari"})

#: Third index of 'zvari' for each quasi-conserved variable (mo_turbdiff_config.f90:62-77),
#: paired with the stencil argument that carries it. The Fortran index appears only here.
CONSERVED_VARIABLES = (
    ("zonal_wind", 1),  # u_m
    ("meridional_wind", 2),  # v_m
    ("liquid_water_potential_temperature", 3),  # tet_l
    ("total_water", 4),  # h2o_g
    ("liquid_water", 5),  # liq
)

experiment_for_turbulence = pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.MCH_ICON_CH2_SMALL],
    ids=lambda d: d.name,
)


def _copy_of(field: gtx.Field, backend) -> gtx.Field:
    """A writable field with the same domain and contents, on the backend under test."""
    return gtx.as_field(field.domain, field.asnumpy().copy(), allocator=backend)


def _nan_like(field: gtx.Field, backend) -> gtx.Field:
    """A writable field with the same domain, filled with NaN.

    Adversarial on purpose: the slot this stands in for is undefined at the entry savepoint, so
    a column the stencil fails to write must not accidentally hold a plausible value.
    """
    return gtx.as_field(field.domain, np.full_like(field.asnumpy(), np.nan), allocator=backend)


def _surface_row(field: gtx.Field, nlev: int, backend) -> gtx.Field:
    """Row 'nlev' of a half-level field as a 2D cell field.

    GT4Py offsets are relative, so a fixed absolute-K input (Fortran 'tkvm(:,ke1)') has to be
    pre-sliced by the caller.
    """
    return gtx.as_field((dims.CellDim,), field.asnumpy()[:, nlev].copy(), allocator=backend)


def _assert_agrees(
    stencil_name: str,
    quantity: str,
    computed: gtx.Field,
    reference: gtx.Field,
    *,
    ivstart: int,
    ivend: int,
    levels: slice = slice(None),
) -> None:
    """Compare one output against the reference under the stencil's declared gate.

    The gate is looked up rather than defaulted: a stencil with no entry in the registry is a
    failure, because a silent 'Exact()' is indistinguishable from one nobody decided on.

    'levels' restricts the comparison to the rows the named stencil is responsible for, so that
    a failure names the program that produced it. It is ignored for the 2D surface fields,
    which have no vertical axis to restrict.
    """
    gate = gate_registry.gate_for(stencil_name)
    window = (slice(ivstart, ivend), levels)[: computed.asnumpy().ndim]
    got = computed.asnumpy()[window]
    want = reference.asnumpy()[window]

    if isinstance(gate, gate_registry.Exact):
        assert np.array_equal(got, want), (
            f"'{stencil_name}' is gated 'Exact' but '{quantity}' differs from ICON: max abs "
            f"{np.nanmax(np.abs(got - want))} over {np.count_nonzero(got != want)} of "
            f"{got.size} values."
        )
    else:
        np.testing.assert_allclose(got, want, rtol=gate.rtol, err_msg=quantity)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_section_1a_writes_exactly_four_slots(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Section 1a) changes 'lays', 'hlp', 'dicke' and 'zvari', and nothing else.

    This is what bounds the four stencils below. Reading the Fortran gives the same answer, but
    the answer depends on the configuration -- a resolved canopy or a different 'imode_*' could
    add a write -- so it is asserted against the capture rather than argued.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="1a", date=date)
    window = slice(before.ivstart(), before.ivend())

    changed = set()
    for name in data_provider.serializer.fields_at_savepoint(before.savepoint):
        entry = np.asarray(data_provider.serializer.read(name, before.savepoint))
        exit_ = np.asarray(data_provider.serializer.read(name, after.savepoint))
        masked = window if entry.ndim >= 2 and entry.shape[0] > before.ivend() else slice(None)
        if not np.array_equal(entry[masked], exit_[masked]):
            changed.add(name)

    assert changed == set(SECTION_1A_OUTPUT_SLOTS)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_compute_surface_transfer_ratios(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="1a", date=date)
    nlev = entry.ke()

    for_momentum = _nan_like(before.tfm(), backend)
    for_scalars = _nan_like(before.tfh(), backend)

    compute_surface_transfer_ratios.with_backend(backend)(
        tvm=entry.tvm(),
        tvh=entry.tvh(),
        tkvm_at_surface=_surface_row(before.tkvm(), nlev, backend),
        tkvh_at_surface=_surface_row(before.tkvh(), nlev, backend),
        tfm=before.tfm(),
        tfh=before.tfh(),
        surface_transfer_ratio_for_momentum=for_momentum,
        surface_transfer_ratio_for_scalars=for_scalars,
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        offset_provider={},
    )

    for quantity, computed, reference in (
        ("lays(:,mom)", for_momentum, after.lays(0)),
        ("lays(:,sca)", for_scalars, after.lays(1)),
    ):
        _assert_agrees(
            "compute_surface_transfer_ratios",
            quantity,
            computed,
            reference,
            ivstart=before.ivstart(),
            ivend=before.ivend(),
        )


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_compute_inverse_layer_depth_and_tke_discretisation_momentum(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="1a", date=date)
    nlev = entry.ke()

    inverse_layer_depth = _copy_of(before.hlp(), backend)
    tke_discretisation_momentum = _copy_of(before.layer_depth(), backend)

    compute_inverse_layer_depth_and_tke_discretisation_momentum.with_backend(backend)(
        hhl=entry.hhl(),
        rhon=before.rhon(),
        inverse_layer_depth=inverse_layer_depth,
        tke_discretisation_momentum=tke_discretisation_momentum,
        inverse_tke_time_step=entry.fr_tke(),
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(nlev),
        offset_provider={dims.Koff.value: dims.KDim},
    )

    for quantity, computed, reference in (
        ("hlp", inverse_layer_depth, after.hlp()),
        ("dicke", tke_discretisation_momentum, after.disc_mom()),
    ):
        _assert_agrees(
            "compute_inverse_layer_depth_and_tke_discretisation_momentum",
            quantity,
            computed,
            reference,
            ivstart=before.ivstart(),
            ivend=before.ivend(),
        )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_compute_gradients_of_conserved_variables(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """One program reproduces section 1a)'s whole 'zvari' profile, surface row included.

    The Fortran writes the surface row in a separate loop from the interior rows because the
    reciprocal depth it divides by lives in a different array there; the arithmetic is the same
    difference quotient. The port therefore keeps one program and selects the depth per row, so
    the vertical boundary is stated once, inside the stencil, next to the Fortran it came from --
    rather than twice, in whatever 'vertical_start'/'vertical_end' each caller happens to pass.

    That makes the boundary itself part of what this test checks. The output is allocated as a
    COPY OF THE ENTRY VARIABLES and compared over the whole column, so the comparison distinguishes
    all three ways the row selection can be wrong: the surface row taking the geometric depth, the
    last interior row taking the Prandtl-layer depth, and the model top being written at all.

    The scale factors -- 'hlp' and 'lays' -- are taken from the EXIT savepoint rather than from the
    two stencils that produce them, so that a defect there cannot travel into this test and a
    failure here means this translation is wrong.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="1a", date=date)
    nlev = entry.ke()
    ivstart, ivend = before.ivstart(), before.ivend()

    variables = {
        name: before.conserved_variable(component) for name, component in CONSERVED_VARIABLES
    }
    # The gradients replace the variables in the Fortran's own storage, so allocating each output
    # as a copy of its input is what reproduces the Fortran's initial state -- and keeping the two
    # buffers apart is what makes the claim that this is not a recurrence testable rather than
    # assumed.
    gradients = {f"{name}_gradient": _copy_of(field, backend) for name, field in variables.items()}

    compute_vertical_gradients_of_conserved_variables.with_backend(backend)(
        **variables,
        **gradients,
        inverse_layer_depth=after.hlp(),
        surface_transfer_ratio_for_momentum=after.lays(0),
        surface_transfer_ratio_for_scalars=after.lays(1),
        nlev=gtx.int32(nlev),
        horizontal_start=gtx.int32(ivstart),
        horizontal_end=gtx.int32(ivend),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(nlev + 1),
        offset_provider={dims.Koff.value: dims.KDim},
    )

    for name, component in CONSERVED_VARIABLES:
        quantity = f"zvari(:,:,{component}) [{name}]"
        _assert_agrees(
            "compute_vertical_gradients_of_conserved_variables",
            quantity,
            gradients[f"{name}_gradient"],
            after.vertical_gradient(component),
            ivstart=ivstart,
            ivend=ivend,
            levels=slice(0, nlev + 1),
        )
        # Row 0 is asserted against the INPUT as well as against the reference above. The two
        # agree only because ICON leaves that row alone, which is the statement being made.
        assert np.array_equal(
            gradients[f"{name}_gradient"].asnumpy()[ivstart:ivend, 0],
            variables[name].asnumpy()[ivstart:ivend, 0],
        ), f"the model top of '{quantity}' was written; the Fortran loop starts at 'k = 2'."


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_gradients_at_the_surface_do_not_read_the_geometric_depth(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The surface row uses the Prandtl-layer depth and never touches 'hlp' there.

    The merged program relies on 'concat_where' evaluating only the branch a row selects. That is
    a property of GT4Py's domain inference, not of the expression, so it is measured rather than
    assumed: 'hlp' is poisoned with NaN on the surface row -- where section 1a) never defines it
    anyway -- and the surface gradients must stay bit-exact. If a backend ever computed the false
    branch over the whole domain and selected afterwards, every value here would be NaN.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="1a", date=date)
    nlev = entry.ke()
    ivstart, ivend = before.ivstart(), before.ivend()

    poisoned = after.hlp().asnumpy().copy()
    poisoned[:, nlev] = np.nan
    inverse_layer_depth = gtx.as_field(after.hlp().domain, poisoned, allocator=backend)

    variables = {
        name: before.conserved_variable(component) for name, component in CONSERVED_VARIABLES
    }
    gradients = {f"{name}_gradient": _copy_of(field, backend) for name, field in variables.items()}

    compute_vertical_gradients_of_conserved_variables.with_backend(backend)(
        **variables,
        **gradients,
        inverse_layer_depth=inverse_layer_depth,
        surface_transfer_ratio_for_momentum=after.lays(0),
        surface_transfer_ratio_for_scalars=after.lays(1),
        nlev=gtx.int32(nlev),
        horizontal_start=gtx.int32(ivstart),
        horizontal_end=gtx.int32(ivend),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(nlev + 1),
        offset_provider={dims.Koff.value: dims.KDim},
    )

    for name, component in CONSERVED_VARIABLES:
        _assert_agrees(
            "compute_vertical_gradients_of_conserved_variables",
            f"zvari(:,ke1,{component}) [{name}] with 'hlp(:,ke1)' poisoned",
            gradients[f"{name}_gradient"],
            after.vertical_gradient(component),
            ivstart=ivstart,
            ivend=ivend,
            levels=slice(nlev, nlev + 1),
        )
