# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 1a) of 'turbdiff', the vertical gradients.

The oracle is serialized ICON, not a hand-written reference: the section is run on the fields
of 'turbdiff-0-exit' and compared against 'turbdiff-1a-exit', for all four timesteps of
exp.mch_icon-ch2_small. A numpy re-implementation would only restate the translation and would
agree with it for the same reason it agrees with itself.

THE OUTPUT SET IS ESTABLISHED, NOT ASSUMED. Section 1a) writes exactly four storage slots --
'lays', 'hlp', 'dicke' and 'zvari' -- and 'test_section_1a_writes_exactly_four_slots' asserts
that by comparing every serialized field across the two savepoints. That is what fixes the
scope of the section; anything else that changed would mean it does more than the one program
below does.

The two conventions every comparison below follows -- masked to 'ivstart:ivend', and each output
field allocated as a copy of its entry state -- are stated once in 'tests/turbulence/utils.py'.
What they buy here: the rows the Fortran leaves alone -- the model top of 'hlp', 'dicke' and the
five gradients, and the surface row of 'hlp' and 'dicke' -- are asserted to be untouched rather
than ignored. 'lays' is undefined before this section, so it is allocated NaN-filled instead,
which makes an unwritten column inside the window a failure rather than a coincidence.

ONE PROGRAM SINCE THE STENCIL MERGE, AND WHAT THAT COST THIS FILE. Section 1a) used to be three
'@gtx.program's -- the surface transfer ratios, the reciprocal layer depth with the TKE
discretisation momentum, and the gradients -- and this file ran each of them separately. They are
now three statements of 'compute_vertical_gradients_of_conserved_variables', so one call produces
all nine outputs and every test below reads that one run.

The assertions are unchanged and still one per output, so a failure still names the Fortran
quantity. One thing genuinely weakened, and it is recorded rather than glossed: the gradient test
used to take 'hlp' and 'lays' from the EXIT savepoint, so that a defect in the two programs that
produce them could not travel into it. Inside one program that isolation is not available -- the
scale factors are computed by the statements above the gradients. What replaces it is that 'hlp'
and 'lays' are asserted against ICON in their own right, bit for bit, by the two tests before it;
a defect there fails those first. Attribution is what got worse, not coverage.

'compute_vertical_gradients_of_conserved_variables' selects its reciprocal depth per row with
'concat_where', which the embedded backend cannot execute (see 'model/testing/filters.py'), so
every test that runs the program carries 'uses_concat_where' and xfails there. The compiled
backends still assert the whole gradient profile bit-exactly, and the poison test pins the row
selection itself. Why the surface row is not a program of its own is written down in the package
README, section "Boundary rows".
"""

from __future__ import annotations

from typing import NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_vertical_gradients_of_conserved_variables import (
    compute_vertical_gradients_of_conserved_variables,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


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


class Section1a(NamedTuple):
    """One timestep of section 1a): the savepoints, the bounds and the nine computed outputs."""

    entry: sb.IconTurbdiffEntrySavepoint
    before: sb.IconTurbdiffSectionSavepoint
    after: sb.IconTurbdiffSectionSavepoint
    nlev: int
    #: Half-open range of columns 'turbdiff' computed; everything else is untouched memory with
    #: plausible values, so every comparison below is masked with it.
    columns: slice
    surface_transfer_ratio_for_momentum: gtx.Field
    surface_transfer_ratio_for_scalars: gtx.Field
    inverse_layer_depth: gtx.Field
    tke_discretisation_momentum: gtx.Field
    #: The five gradients, keyed by the stencil argument name without the '_gradient' suffix.
    gradients: dict[str, gtx.Field]
    #: The five conserved variables the gradients were formed from, same keys.
    variables: dict[str, gtx.Field]


def _run_section_1a(data_provider, date: str, backend, *, poison: str | None = None) -> Section1a:
    """Run section 1a) on the 'turbdiff-0-exit' state of one timestep.

    ONE VERTICAL BOUND IS STATED HERE, WHERE THERE USED TO BE TWO. The helper passes
    'vertical_end = nlev + 1', the gradients' range; the 'hlp'/'dicke' statement inside the
    program stops at 'vertical_end - 1'. What still constrains that bound from outside is the
    comparison of 'hlp' and 'dicke' over the WHOLE column against the reference: the surface row
    of both is one the Fortran never writes, so a statement that ran one row further would fail
    that comparison -- it would overwrite a row the reference leaves at its entry value.

    Args:
        data_provider: The serialized archive.
        date: One of 'utils.TURBDIFF_DATES'.
        backend: The backend under test.
        poison: 'surface-layer-depth' puts NaN into the surface row of the 'hlp' output buffer,
            which is a row the program must neither write nor read. See
            'test_gradients_at_the_surface_do_not_read_the_geometric_depth'.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="1a", date=date)
    nlev = entry.ke()

    # 'lays' is undefined before this section, so NaN rather than a copy; the other three
    # storages carry their entry state in, which is what makes the untouched rows assertable.
    for_momentum = utils.nan_like(before.tfm(), backend)
    for_scalars = utils.nan_like(before.tfh(), backend)
    inverse_layer_depth = utils.copy_of(before.hlp(), backend)
    tke_discretisation_momentum = utils.copy_of(before.layer_depth(), backend)

    if poison == "surface-layer-depth":
        values = inverse_layer_depth.asnumpy().copy()
        values[:, nlev] = np.nan
        inverse_layer_depth = gtx.as_field(inverse_layer_depth.domain, values, allocator=backend)

    variables = {
        name: before.conserved_variable(component) for name, component in CONSERVED_VARIABLES
    }
    # The gradients replace the variables in the Fortran's own storage, so allocating each output
    # as a copy of its input is what reproduces the Fortran's initial state -- and keeping the two
    # buffers apart is what makes the claim that this is not a recurrence testable rather than
    # assumed.
    gradients = {name: utils.copy_of(field, backend) for name, field in variables.items()}

    compute_vertical_gradients_of_conserved_variables.with_backend(backend)(
        tvm=entry.tvm(),
        tvh=entry.tvh(),
        tkvm_at_surface=utils.surface_row(before.tkvm(), nlev, backend),
        tkvh_at_surface=utils.surface_row(before.tkvh(), nlev, backend),
        tfm=before.tfm(),
        tfh=before.tfh(),
        hhl=entry.hhl(),
        rhon=before.rhon(),
        inverse_tke_time_step=entry.fr_tke(),
        **variables,
        nlev=gtx.int32(nlev),
        surface_transfer_ratio_for_momentum=for_momentum,
        surface_transfer_ratio_for_scalars=for_scalars,
        inverse_layer_depth=inverse_layer_depth,
        tke_discretisation_momentum=tke_discretisation_momentum,
        **{f"{name}_gradient": field for name, field in gradients.items()},
        horizontal_start=gtx.int32(before.ivstart()),
        horizontal_end=gtx.int32(before.ivend()),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(nlev + 1),
        offset_provider={dims.Koff.value: dims.KDim},
    )
    return Section1a(
        entry=entry,
        before=before,
        after=after,
        nlev=nlev,
        columns=slice(before.ivstart(), before.ivend()),
        surface_transfer_ratio_for_momentum=for_momentum,
        surface_transfer_ratio_for_scalars=for_scalars,
        inverse_layer_depth=inverse_layer_depth,
        tke_discretisation_momentum=tke_discretisation_momentum,
        gradients=gradients,
        variables=variables,
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_1a_writes_exactly_four_slots(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Section 1a) changes 'lays', 'hlp', 'dicke' and 'zvari', and nothing else.

    This is what bounds the program below; 'utils.fields_that_changed' says why the output set
    is measured against the capture rather than read off the Fortran.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="1a", date=date)

    assert utils.fields_that_changed(data_provider, before, after) == SECTION_1A_OUTPUT_SLOTS


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_surface_transfer_ratios(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The two surface transfer ratios, from the statement with no vertical axis at all."""
    run = _run_section_1a(data_provider, date, backend)

    for quantity, computed, reference in (
        ("lays(:,mom)", run.surface_transfer_ratio_for_momentum, run.after.lays(0)),
        ("lays(:,sca)", run.surface_transfer_ratio_for_scalars, run.after.lays(1)),
    ):
        utils.assert_agrees_with_icon(
            "compute_vertical_gradients_of_conserved_variables",
            quantity,
            computed,
            reference,
            columns=run.columns,
        )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_inverse_layer_depth_and_tke_discretisation_momentum(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'hlp' and 'dicke' over the whole column, so the two rows they do not write are checked.

    The comparison is not restricted to rows 1..'nlev'-1: both buffers carry their entry state
    in, and the model top and the surface row must come out as this section found them. That is
    what constrains the statement's 'vertical_end - 1' from outside the stencil.
    """
    run = _run_section_1a(data_provider, date, backend)

    for quantity, computed, reference in (
        ("hlp", run.inverse_layer_depth, run.after.hlp()),
        ("dicke", run.tke_discretisation_momentum, run.after.disc_mom()),
    ):
        utils.assert_agrees_with_icon(
            "compute_vertical_gradients_of_conserved_variables",
            quantity,
            computed,
            reference,
            columns=run.columns,
        )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_gradients_of_conserved_variables(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The whole 'zvari' profile, surface row included.

    The Fortran writes the surface row in a separate loop from the interior rows because the
    reciprocal depth it divides by lives in a different array there; the arithmetic is the same
    difference quotient. The port therefore keeps one expression and selects the depth per row, so
    the vertical boundary is stated once, inside the stencil, next to the Fortran it came from --
    rather than in whatever 'vertical_start'/'vertical_end' each caller happens to pass.

    That makes the boundary itself part of what this test checks. The output is allocated as a
    COPY OF THE ENTRY VARIABLES and compared over the whole column, so the comparison distinguishes
    all three ways the row selection can be wrong: the surface row taking the geometric depth, the
    last interior row taking the Prandtl-layer depth, and the model top being written at all.
    """
    run = _run_section_1a(data_provider, date, backend)

    for name, component in CONSERVED_VARIABLES:
        quantity = f"zvari(:,:,{component}) [{name}]"
        utils.assert_agrees_with_icon(
            "compute_vertical_gradients_of_conserved_variables",
            quantity,
            run.gradients[name],
            run.after.vertical_gradient(component),
            columns=run.columns,
            levels=slice(0, run.nlev + 1),
        )
        # Row 0 is asserted against the INPUT as well as against the reference above. The two
        # agree only because ICON leaves that row alone, which is the statement being made.
        assert np.array_equal(
            run.gradients[name].asnumpy()[run.columns, 0],
            run.variables[name].asnumpy()[run.columns, 0],
        ), f"the model top of '{quantity}' was written; the Fortran loop starts at 'k = 2'."


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_gradients_at_the_surface_do_not_read_the_geometric_depth(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The surface row uses the Prandtl-layer depth and never touches 'hlp' there.

    The merged program relies on 'concat_where' evaluating only the branch a row selects. That is
    a property of GT4Py's domain inference, not of the expression, so it is measured rather than
    assumed: 'hlp' is poisoned with NaN on the surface row -- where section 1a) neither writes it
    nor defines it -- and the surface gradients must stay bit-exact. If a backend ever computed
    the false branch over the whole domain and selected afterwards, every value here would be NaN.

    THE POISON MOVED FROM AN INPUT TO AN OUTPUT BUFFER when the three programs became one, and
    the property it pins did not. 'hlp' used to be handed in from the exit savepoint and could be
    poisoned before the call; it is now written by the statement above the gradients, over rows
    1..'nlev'-1 only. Poisoning the surface row of that buffer therefore still presents the
    gradient statement with a NaN in exactly the row its fallback branch must not evaluate -- and
    it additionally asserts that the 'hlp' statement does not write that row either, since a NaN
    that was overwritten would prove nothing.
    """
    run = _run_section_1a(data_provider, date, backend, poison="surface-layer-depth")

    assert np.isnan(run.inverse_layer_depth.asnumpy()[run.columns, run.nlev]).all(), (
        "the 'hlp' statement wrote the surface row, so the poison below was overwritten and "
        "this test no longer measures the row selection it claims to."
    )
    for name, component in CONSERVED_VARIABLES:
        utils.assert_agrees_with_icon(
            "compute_vertical_gradients_of_conserved_variables",
            f"zvari(:,ke1,{component}) [{name}] with 'hlp(:,ke1)' poisoned",
            run.gradients[name],
            run.after.vertical_gradient(component),
            columns=run.columns,
            levels=slice(run.nlev, run.nlev + 1),
        )
