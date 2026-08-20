# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test of the muphys ComponentState adapter: pure input mapping, no copies."""

import types

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import state as muphys_state
from icon4py.model.common.metrics import metrics_attributes


class _StubFieldSource:
    def __init__(self, fields):
        self._fields = fields

    def get(self, name, *args, **kwargs):
        return self._fields[name]


def test_as_component_input_maps_the_facade_without_copies():
    dz = object()
    state = muphys_state.State(metrics=_StubFieldSource({metrics_attributes.DDQZ_Z_FULL: dz}))
    tracers = types.SimpleNamespace(qv="QV", qc="QC", qi="QI", qr="QR", qs="QS", qg="QG")
    entry = types.SimpleNamespace(ta="TA", pressure="P", rho="RHO", tracers=tracers)

    state.collect_inputs(entry)
    inputs = state.as_component_input()

    assert inputs == {
        "dz": dz,
        "te": "TA",
        "p": "P",
        "rho": "RHO",
        "qv": "QV",
        "qc": "QC",
        "qi": "QI",
        "qr": "QR",
        "qs": "QS",
        "qg": "QG",
    }
    assert inputs["dz"] is dz  # the metrics field itself, not a copy


def test_as_component_input_requires_collect_inputs_first():
    state = muphys_state.State(metrics=_StubFieldSource({metrics_attributes.DDQZ_Z_FULL: object()}))
    with pytest.raises(RuntimeError, match="collect_inputs"):
        state.as_component_input()
