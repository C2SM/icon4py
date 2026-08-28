# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared plumbing for the turbulence section datatests.

Every section of this port is validated the same way -- the serialized ICON run is the oracle,
one savepoint supplies the inputs and the next one the expected outputs -- so the mechanics of
setting that up belong here and not in each section's test module. What is here is the
mechanics only; what a section computes, what it is allowed to write and why, stays in the
module that tests it.

Import it as a module, so a call site says where the helper comes from:

    from .. import utils

    @pytest.mark.datatest
    @utils.experiment_for_turbulence
    @pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
    def test_compute_something(date, *, data_provider, backend):
        before = data_provider.from_savepoint_turbdiff_section(section="1a", date=date)
        after = data_provider.from_savepoint_turbdiff_section(section="1b", date=date)
        columns = slice(before.ivstart(), before.ivend())

        computed = utils.copy_of(before.hlp(), backend)
        compute_something.with_backend(backend)(..., some_output=computed, ...)

        utils.assert_agrees_with_icon(
            "compute_something", "hlp", computed, after.hlp(), columns=columns
        )

Two conventions run through all of it, and a new section is expected to keep them:

EVERY COMPARISON IS MASKED TO 'ivstart:ivend'. The Fortran hooks write the whole 'nproma' slab
but the schemes only loop over that window. What lies outside is untouched memory holding
plausible values rather than NaN, so an unmasked comparison fails looking exactly like a physics
bug. 'assert_agrees_with_icon' takes the window; it does not guess it.

AN OUTPUT FIELD STARTS AS ITS ENTRY STATE, and is compared over the whole slab. The rows a
section does not write must then come out unchanged, which turns a wrong vertical domain into a
failed assertion instead of an invisible one. Zero-filling would make those rows differ for a
reason that says nothing about the stencil. 'copy_of' and 'copy_of_raw_field' are the two ways
to get that starting state; 'nan_like' is for the one case where there is none.
"""

from __future__ import annotations

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, serialbox as sb

from . import gate_registry


__all__ = [
    "TURBDIFF_DATES",
    "assert_agrees_with_icon",
    "copy_of",
    "copy_of_raw_field",
    "experiment_for_turbulence",
    "fields_that_changed",
    "nan_like",
    "surface_row",
]


#: The four timesteps 'exp.mch_icon-ch2_small' serializes: the serialization window is the last
#: four of its six steps, and each scheme is called once per step per block. 'turbtran' and
#: 'vertdiff' serialize the same four, so their section tests parametrize over this too.
TURBDIFF_DATES = (
    "2020-12-10T06:01:00.000",
    "2020-12-10T06:01:20.000",
    "2020-12-10T06:01:40.000",
    "2020-12-10T06:02:00.000",
)

#: The experiment every turbulence datatest is validated against, as a decorator to apply to each
#: test. It parametrizes 'experiment_description', which is what the 'data_provider' fixture
#: resolves the archive from; a test that omits it gets whatever experiment the fixture defaults
#: to and silently reads someone else's data.
experiment_for_turbulence = pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.MCH_ICON_CH2_SMALL],
    ids=lambda d: d.name,
)


# --------------------------------------------------------------------------- output buffers ---


def copy_of(field: gtx.Field, backend: gtx_typing.Backend | None) -> gtx.Field:
    """A writable field with the same domain and contents, on the backend under test.

    The default way to allocate an output: it reproduces the state the Fortran storage was in
    when the section started, so the levels the section leaves alone can be asserted to be
    untouched instead of being excluded from the comparison.

    'field' comes from a named savepoint accessor. When the reader refuses to name the slot at
    the entry savepoint -- the usual case for a slot the section under test is the first to
    write -- use 'copy_of_raw_field' instead.
    """
    return gtx.as_field(field.domain, field.asnumpy().copy(), allocator=backend)


def copy_of_raw_field(
    savepoint: sb.IconTurbulenceSavepoint, name: str, backend: gtx_typing.Backend | None
) -> gtx.Field:
    """'copy_of' for a storage slot the savepoint reader will not name, as a (Cell, K) field.

    'IconTurbdiffSectionSavepoint' gives every reused Fortran array one accessor per role and
    each accessor raises outside the sections where its role holds. A slot the section under test
    is the first to write therefore has no accessor at its own entry savepoint -- 'frh' is
    'thermal_forcing()' from section 1b) on, and nothing at all at 1a), which is where section
    1b)'s test has to read it. 'raw_field' is that reader's documented escape hatch and this
    wraps it: the raw buffer is the untruncated 'nproma' slab, so it is cut down to the grid's
    cells here, which is what the named accessors do too.

    'raw_field' hands back an array in the BACKEND'S array namespace, so on a GPU backend it
    is a 'cupy.ndarray' and 'np.asarray' on it raises rather than copying ("Implicit
    conversion to a NumPy array is not allowed"). It is brought to the host explicitly here,
    and the transfer back to the device is left to 'gtx.as_field(..., allocator=backend)'.

    Args:
        savepoint: The section's entry savepoint.
        name: The serialized name, scheme prefix included ('td_frh', 'vd_frm').
        backend: The backend under test.
    """
    buffer = data_alloc.as_numpy(savepoint.raw_field(name))[: savepoint.sizes[dims.CellDim]]
    return gtx.as_field((dims.CellDim, dims.KDim), np.ascontiguousarray(buffer), allocator=backend)


def nan_like(field: gtx.Field, backend: gtx_typing.Backend | None) -> gtx.Field:
    """A writable field with the same domain, filled with NaN.

    For an output slot that is undefined at the entry savepoint, where the copy convention has
    nothing to copy. Adversarial on purpose: a column the stencil fails to write must not
    accidentally hold a plausible value.

    It costs what 'copy_of' buys -- rows outside the stencil's vertical domain come out NaN and
    can be compared against nothing -- so reach for it only when the entry state really is
    undefined, not merely inconvenient to read.
    """
    return gtx.as_field(field.domain, np.full_like(field.asnumpy(), np.nan), allocator=backend)


def surface_row(field: gtx.Field, level: int, backend: gtx_typing.Backend | None) -> gtx.Field:
    """Row 'level' of a 2D field as a 1D cell field.

    GT4Py offsets are relative, so an input the Fortran reads at a fixed absolute K -- 'tkvm(:,ke1)'
    and its like -- has to be pre-sliced by the caller and passed as a cell field. 'level' is the
    zero-based row, so the surface half level 'ke1' is 'entry.ke()'.
    """
    return gtx.as_field((dims.CellDim,), field.asnumpy()[:, level].copy(), allocator=backend)


# ------------------------------------------------------------------------------ comparison ---


def assert_agrees_with_icon(
    stencil_name: str,
    quantity: str,
    computed: gtx.Field | np.ndarray,
    reference: gtx.Field | np.ndarray,
    *,
    columns: slice = slice(None),
    levels: slice = slice(None),
) -> None:
    """Compare one output against the ICON reference under the stencil's declared gate.

    The gate is looked up rather than defaulted: a stencil with no entry in 'gate_registry' is a
    failure, because a silent 'Exact()' is indistinguishable from one nobody decided on. Add the
    entry -- 'Exact()' unless a measurement says otherwise -- before writing the test that needs
    it.

    Args:
        stencil_name: The stencil under test, as keyed in 'gate_registry.GATES'.
        quantity: What is being compared, for the failure message. Name it as the Fortran does
            ('hlp', 'zvari(:,:,3) [tet_l]'), since that is what a failure has to be traced back to.
        computed: The field the stencil wrote, or a plain array if the caller sliced it already.
        reference: The same quantity from the exit savepoint.
        columns: The computed columns, 'slice(sp.ivstart(), sp.ivend())'. It defaults to
            everything only so that an already-masked array can be passed; a comparison against
            an unmasked slab needs it.
        levels: The rows the named stencil is responsible for, so that a failure names the
            program that produced it. Ignored for 1D cell fields, which have no vertical axis.

    Bit-exactness is asserted with 'array_equal', under which NaN never equals NaN: a NaN on both
    sides means the stencil wrote nothing there and the reference says nothing either, which is
    not agreement.
    """
    gate = gate_registry.gate_for(stencil_name)
    got = _as_array(computed)
    want = _as_array(reference)
    window = (columns, levels)[: got.ndim]
    got, want = got[window], want[window]

    if isinstance(gate, gate_registry.Exact):
        assert np.array_equal(got, want), (
            f"'{stencil_name}' is gated 'Exact' but '{quantity}' differs from ICON: max abs "
            f"{np.nanmax(np.abs(got - want))} over {np.count_nonzero(got != want)} of "
            f"{got.size} values."
        )
    else:
        np.testing.assert_allclose(got, want, rtol=gate.rtol, atol=0.0, err_msg=quantity)


def fields_that_changed(
    data_provider: sb.IconSerialDataProvider,
    before: sb.IconTurbulenceSavepoint,
    after: sb.IconTurbulenceSavepoint,
) -> frozenset[str]:
    """The serialized names that differ between two savepoints, over the computed columns.

    This is how a section's output set is established rather than assumed, and it is what bounds
    the stencils a section test may contain. Reading the Fortran gives the same answer, but the
    answer depends on the configuration -- a resolved canopy, a different 'imode_*' or a namelist
    switch that is not serialized could add a write -- so it is measured against the capture.

    Every name serialized at 'before' is read at both savepoints and compared over
    'ivstart:ivend'. A buffer with fewer rows than that is compared whole instead of being sliced
    down to nothing: the '(1,1)' dummies and the scalars would otherwise be unable to differ.

    NaN never compares equal to NaN, deliberately -- a slot holding NaN at both boundaries has
    not been shown to be untouched.

    IT CANNOT BE USED ACROSS 'turbdiff-exit'. Both savepoints have to come from the same
    serialization hook. The section hook is one Fortran statement called fifteen times, so every
    section pair sees one field table; 'turbdiff-exit' is a different hook with a different table,
    and the two differ in ten names -- five 'turbdiff' locals ('td_hor_scale', 'td_layr',
    'td_lays', 'td_nvor', 'td_xri') that never reach the interface and so are not at the exit
    hook, and five interface fields ('td_gz0', 'td_tvm', 'td_tvh', 'td_tkred_sfc',
    'td_tkred_sfc_h') that the section hook does not carry.

    Given such a pair this function raises 'SerialboxError' on the first name the exit hook does
    not have, which is the intended behaviour: the alternative -- skipping what one side lacks --
    would return an answer that silently says nothing about ten fields, and "the output set is
    exactly this" is a claim that must not be quietly narrowed. Section 11) is the only pair in
    the port that crosses the exit hook; it does the skipping explicitly, in
    '_fields_that_changed_across_the_exit_hook', and accounts for each of the ten names it drops.
    """
    ivstart, ivend = before.ivstart(), before.ivend()
    window = slice(ivstart, ivend)
    changed = set()
    for name in data_provider.serializer.fields_at_savepoint(before.savepoint):
        entry = np.asarray(data_provider.serializer.read(name, before.savepoint))
        exit_ = np.asarray(data_provider.serializer.read(name, after.savepoint))
        masked = window if entry.ndim >= 2 and entry.shape[0] > ivend else slice(None)
        if not np.array_equal(entry[masked], exit_[masked]):
            changed.add(name)
    return frozenset(changed)


def _as_array(field: gtx.Field | np.ndarray) -> np.ndarray:
    """The values of a GT4Py field, or of a plain array, as host memory.

    'data_alloc.as_numpy' rather than 'np.asarray' because on a GPU backend the array may be a
    'cupy.ndarray', which refuses implicit conversion -- see 'copy_of_raw_field'.
    """
    return data_alloc.as_numpy(field)
