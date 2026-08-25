# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the tmx ComponentState adapter: input mapping + derived inputs."""

import types

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import data as tmx_data, state as tmx_state
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.states import tracer_states
from icon4py.model.common.utils import data_allocation as data_alloc


class _StubFieldSource:
    """Minimal FieldSource stand-in: serves pre-built fields by attribute name."""

    def __init__(self, fields):
        self._fields = fields

    def get(self, name, *args, **kwargs):
        return self._fields[name]


def _tracer_state(grid, *, qv: float = 0.0) -> tracer_states.TracerState:
    ck = lambda value: data_alloc.constant_field(grid, value, dims.CellDim, dims.KDim)  # noqa: E731
    return tracer_states.TracerState(
        qv=ck(qv), qc=ck(0.0), qi=ck(0.0), qr=ck(0.0), qs=ck(0.0), qg=ck(0.0)
    )


def _entry_stub(grid, *, qv: float = 1e-3, rho: float = 1.2):
    """EntryState stand-in: real fields where the adapter runs stencils, sentinels elsewhere."""
    return types.SimpleNamespace(
        ta="TA",
        tv="TV",
        pressure="P",
        pressure_ifc="P_IFC",
        u="U",
        v="V",
        w="W",
        rho=data_alloc.constant_field(grid, rho, dims.CellDim, dims.KDim),
        tracers=_tracer_state(grid, qv=qv),
    )


def _tmx_state(grid, **kwargs) -> tmx_state.State:
    metrics = _StubFieldSource(
        {
            metrics_attributes.DDQZ_Z_FULL: data_alloc.constant_field(
                grid, 100.0, dims.CellDim, dims.KDim
            ),
        }
    )
    return tmx_state.State(grid=grid, metrics=metrics, backend=None, **kwargs)


def test_collect_inputs_derives_and_maps_the_contract():
    grid = simple.simple_grid()
    state = _tmx_state(grid)
    entry = _entry_stub(grid, qv=1e-3, rho=1.2)

    state.collect_inputs(entry)
    inputs = state.as_component_input()

    # exactly the component contract, nothing more or less
    assert set(inputs) == set(tmx_data.INPUTS_PROPERTIES)
    # derived inputs: air_mass = rho * dz; cv_air is a positive heat capacity
    np.testing.assert_allclose(state.air_mass.asnumpy(), 1.2 * 100.0, rtol=1e-12)
    assert (state.cv_air.asnumpy() > 0).all()
    # facade fields pass through untouched (pointers, no copies)
    assert inputs["temperature"] == "TA"
    assert inputs["w"] == "W"
    assert inputs["qv"] is entry.tracers.qv
    assert inputs["air_mass"] is state.air_mass


def test_as_component_input_requires_collect_inputs_first():
    with pytest.raises(RuntimeError, match="collect_inputs"):
        _tmx_state(simple.simple_grid()).as_component_input()
