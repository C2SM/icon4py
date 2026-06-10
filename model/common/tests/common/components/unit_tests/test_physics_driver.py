# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime

import pytest

from icon4py.model.common.components.physics_driver import (
    ForcingMode,
    PhysicsDriver,
    PhysicsProcess,
    ProcessTimeControl,
)
from icon4py.model.common.states.model import FieldMetaData


def test_field_metadata_accepts_kind() -> None:
    meta: FieldMetaData = {
        "standard_name": "tend_temperature",
        "units": "K s-1",
        "kind": "tendency",
    }
    assert meta["kind"] == "tendency"


def test_forcing_mode_values() -> None:
    assert ForcingMode.DIAGNOSTIC.value == 0
    assert ForcingMode.APPLY.value == 1
    assert ForcingMode.DIAGNOSTIC is not ForcingMode.APPLY
    assert len(ForcingMode) == 2


_T0 = datetime.datetime(2024, 1, 1, 0, 0, 0)
_DT = datetime.timedelta(seconds=300)  # 5-min physics interval


def _tc(
    interval: datetime.timedelta = _DT,
    start: datetime.datetime = _T0,
    end: datetime.datetime = _T0 + datetime.timedelta(days=1),
    mode: ForcingMode = ForcingMode.APPLY,
) -> ProcessTimeControl:
    return ProcessTimeControl(interval=interval, start_date=start, end_date=end, forcing_mode=mode)


class TestProcessTimeControl:
    def test_is_enabled_true_when_interval_positive(self) -> None:
        assert _tc(interval=_DT).is_enabled() is True

    def test_is_enabled_false_when_interval_zero(self) -> None:
        assert _tc(interval=datetime.timedelta(0)).is_enabled() is False

    def test_is_active_false_when_disabled(self) -> None:
        assert _tc(interval=datetime.timedelta(0)).is_active(_T0) is False

    def test_is_in_window_at_start_is_true(self) -> None:
        assert _tc().is_in_window(_T0) is True

    def test_is_in_window_at_end_is_false(self) -> None:
        end = _T0 + datetime.timedelta(hours=1)
        assert _tc(end=end).is_in_window(end) is False

    def test_is_in_window_before_start_is_false(self) -> None:
        assert _tc().is_in_window(_T0 - datetime.timedelta(seconds=1)) is False

    def test_is_in_window_inside_is_true(self) -> None:
        assert _tc().is_in_window(_T0 + datetime.timedelta(hours=12)) is True

    def test_is_active_at_start_is_true(self) -> None:
        assert _tc().is_active(_T0) is True

    def test_is_active_at_one_interval_is_true(self) -> None:
        assert _tc().is_active(_T0 + _DT) is True

    def test_is_active_at_half_interval_is_false(self) -> None:
        assert _tc().is_active(_T0 + _DT / 2) is False

    def test_is_active_before_start_is_false(self) -> None:
        assert _tc().is_active(_T0 - datetime.timedelta(seconds=1)) is False

    def test_is_active_tolerates_microsecond_jitter(self) -> None:
        # 2 intervals + 1 microsecond — should still count as "fire on the boundary"
        jitter = datetime.timedelta(microseconds=1)
        assert _tc().is_active(_T0 + 2 * _DT + jitter) is True

    def test_frozen_dataclass(self) -> None:
        tc = _tc()
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.interval = datetime.timedelta(seconds=1)  # type: ignore[misc]


def test_physics_process_construction() -> None:
    class _DummyComponent:
        inputs_properties = {}
        outputs_properties = {}

        def __call__(self, state, time_step):
            return {}

    proc = PhysicsProcess(
        name="muphys",
        granule=_DummyComponent(),
        time_control=_tc(),
    )
    assert proc.name == "muphys"
    assert proc.granule is not None
    assert proc.time_control.is_enabled()


@dataclasses.dataclass
class RecordingComponent:
    """Stub Component: records calls, returns configured outputs.

    `output_kinds` keys mirror `outputs` keys; values are 'tendency' or
    'diagnostic'.
    """

    outputs: dict[str, object]
    output_kinds: dict[str, str]
    call_count: int = 0
    last_state: dict | None = None
    last_time: datetime.datetime | None = None

    @property
    def inputs_properties(self) -> dict:
        return {}

    @property
    def outputs_properties(self) -> dict:
        return {
            k: {"standard_name": k, "units": "1", "kind": self.output_kinds[k]}
            for k in self.outputs
        }

    def __call__(self, state, time_step):
        self.call_count += 1
        self.last_state = state
        self.last_time = time_step
        return dict(self.outputs)


@dataclasses.dataclass
class RecordingPhysicsState:
    """Stub PhysicsState: records refresh / scatter; returns a fixed dict
    from as_component_input. Implements just enough surface for L2."""

    gather_calls: list = dataclasses.field(default_factory=list)
    scatter_calls: list = dataclasses.field(default_factory=list)

    def gather_from_prognostic(self, prognostic) -> None:
        self.gather_calls.append(prognostic)

    def as_component_input(self) -> dict:
        return {"foo": "bar"}

    def scatter_to_prognostic(self, prognostic, outputs, dt) -> None:
        self.scatter_calls.append((prognostic, outputs, dt))


def test_recording_doubles_record_calls() -> None:
    component = RecordingComponent(
        outputs={"tend_temperature": "T_TEND_VALUE", "pflx": "PFLX_VALUE"},
        output_kinds={"tend_temperature": "tendency", "pflx": "diagnostic"},
    )
    state = RecordingPhysicsState()

    # Simulate what PhysicsDriver would do.
    state.gather_from_prognostic("prog")
    out = component(state.as_component_input(), _T0)
    state.scatter_to_prognostic("prog", out, 300.0)

    assert state.gather_calls == ["prog"]
    assert component.call_count == 1
    assert component.last_state == {"foo": "bar"}  # what as_component_input returned
    assert state.scatter_calls == [("prog", out, 300.0)]


def test_run_invokes_components_in_order() -> None:
    state = RecordingPhysicsState()
    comp_a = RecordingComponent(
        outputs={"tend_temperature": "A"},
        output_kinds={"tend_temperature": "tendency"},
    )
    comp_b = RecordingComponent(
        outputs={"tend_temperature": "B"},
        output_kinds={"tend_temperature": "tendency"},
    )

    driver = PhysicsDriver(
        processes=[
            PhysicsProcess(name="A", granule=comp_a, time_control=_tc()),
            PhysicsProcess(name="B", granule=comp_b, time_control=_tc()),
        ],
        physics_state=state,
    )

    driver.run(prognostic="prog", dt=300.0, now=_T0)

    assert comp_a.call_count == 1
    assert comp_b.call_count == 1
    # B's scatter must follow A's (operator-splitting ordering)
    assert state.scatter_calls[0][1] == {"tend_temperature": "A"}
    assert state.scatter_calls[1][1] == {"tend_temperature": "B"}


def test_disabled_process_is_skipped() -> None:
    state = RecordingPhysicsState()
    comp = RecordingComponent(
        outputs={"tend_temperature": "X"},
        output_kinds={"tend_temperature": "tendency"},
    )
    tc_disabled = _tc(interval=datetime.timedelta(0))

    driver = PhysicsDriver(
        processes=[PhysicsProcess(name="disabled", granule=comp, time_control=tc_disabled)],
        physics_state=state,
    )

    driver.run(prognostic="prog", dt=300.0, now=_T0)

    assert comp.call_count == 0
    assert state.scatter_calls == []


def test_out_of_window_process_does_nothing() -> None:
    state = RecordingPhysicsState()
    comp = RecordingComponent(
        outputs={"tend_temperature": "X"},
        output_kinds={"tend_temperature": "tendency"},
    )
    # Window starts in the future — `now=_T0` is before it.
    future = _T0 + datetime.timedelta(days=1)
    tc = _tc(start=future, end=future + datetime.timedelta(hours=1))

    driver = PhysicsDriver(
        processes=[PhysicsProcess(name="future", granule=comp, time_control=tc)],
        physics_state=state,
    )

    driver.run(prognostic="prog", dt=300.0, now=_T0)

    assert comp.call_count == 0
    assert state.scatter_calls == []


def test_active_call_caches_outputs_and_applies_them() -> None:
    state = RecordingPhysicsState()
    comp = RecordingComponent(
        outputs={"tend_temperature": "FRESH"},
        output_kinds={"tend_temperature": "tendency"},
    )
    driver = PhysicsDriver(
        processes=[PhysicsProcess(name="p", granule=comp, time_control=_tc())],
        physics_state=state,
    )

    driver.run(prognostic="prog", dt=300.0, now=_T0)

    assert comp.call_count == 1
    assert state.scatter_calls == [("prog", {"tend_temperature": "FRESH"}, 300.0)]


def test_inactive_in_window_recycles_cached_outputs() -> None:
    state = RecordingPhysicsState()
    # Component returns "FRESH" the first time, would return "STALE" the second
    # if called — but on the recycle step it MUST NOT be called.
    comp = RecordingComponent(
        outputs={"tend_temperature": "FRESH"},
        output_kinds={"tend_temperature": "tendency"},
    )
    # interval = 2 * dt → process fires every other call.
    interval = 2 * _DT
    tc = _tc(interval=interval)
    driver = PhysicsDriver(
        processes=[PhysicsProcess(name="p", granule=comp, time_control=tc)],
        physics_state=state,
    )

    # Step 1: active (elapsed == 0), compute + cache.
    driver.run(prognostic="prog", dt=_DT.total_seconds(), now=_T0)
    # Step 2: in window, but not active (elapsed == _DT, not a multiple of 2*_DT).
    driver.run(prognostic="prog", dt=_DT.total_seconds(), now=_T0 + _DT)

    # Component invoked once total (compute step only).
    assert comp.call_count == 1
    # But scatter happened twice — once with the fresh tendency, once recycled.
    assert len(state.scatter_calls) == 2
    assert state.scatter_calls[0][1] == {"tend_temperature": "FRESH"}
    assert state.scatter_calls[1][1] == {"tend_temperature": "FRESH"}  # recycled
