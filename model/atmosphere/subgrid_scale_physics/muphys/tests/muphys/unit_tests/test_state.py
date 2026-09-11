# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test of the muphys ComponentState adapter: pure input mapping, no copies."""

import types

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import (
    data as muphys_data,
    state as muphys_state,
)
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.states import model


class _StubFieldSource:
    def __init__(self, fields):
        self._fields = fields

    def get(self, name, *args, **kwargs):
        return self._fields[name]


def test_as_component_input_maps_the_facade_without_copies():
    dz = object()
    state = muphys_state.State(metrics=_StubFieldSource({metrics_attributes.DDQZ_Z_FULL: dz}))
    tracers = types.SimpleNamespace(qv="QV", qc="QC", qi="QI", qr="QR", qs="QS", qg="QG")
    entry = types.SimpleNamespace(
        diagnostics=types.SimpleNamespace(temperature="TA", pressure="P"),
        rho="RHO",
        tracers=tracers,
    )

    inputs = state.as_component_input(entry)

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


def test_diagnostic_outputs_declare_dims():
    """Every non-tendency output must declare dims -- the DiagnosticsStore allocates from it."""
    for name, props in muphys_data.OUTPUTS_PROPERTIES.items():
        if props.kind == model.FieldKind.TENDENCY:
            continue
        assert props.dims is not None, f"diagnostic output '{name}' must declare dims"
