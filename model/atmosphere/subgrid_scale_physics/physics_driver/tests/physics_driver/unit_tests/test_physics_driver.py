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

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.physics_driver import (
    PhysicsDriver,
    PhysicsProcess,
)
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.process_time_control import (
    ProcessTimeControl,
)
from icon4py.model.common.components.component_state import ComponentState
from icon4py.model.common.states.model import FieldMetaData


def test_field_metadata_accepts_kind() -> None:
    meta: FieldMetaData = {
        "standard_name": "tend_temperature",
        "units": "K s-1",
        "kind": "tendency",
    }
    assert meta["kind"] == "tendency"


_T0 = datetime.datetime(2024, 1, 1, 0, 0, 0)
_DT = datetime.timedelta(seconds=300)  # 5-min physics interval


def _tc(
    interval: datetime.timedelta = _DT,
    start: datetime.datetime = _T0,
    end: datetime.datetime = _T0 + datetime.timedelta(days=1),
    enable_process: bool = True,
) -> ProcessTimeControl:
    return ProcessTimeControl(
        interval=interval,
        start_date=start,
        end_date=end,
        enable_process=enable_process,
    )


class TestProcessTimeControl:
    def test_enable_process_defaults_true(self) -> None:
        assert _tc().enable_process is True

    def test_is_active_false_when_disabled(self) -> None:
        assert _tc(enable_process=False).is_active(_T0) is False

    def test_is_active_false_when_interval_zero(self) -> None:
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

    def test_is_active_requires_exact_interval_multiple(self) -> None:
        # Fires only at an exact integer multiple of the interval.
        assert _tc().is_active(_T0 + 2 * _DT) is True
        # 1 microsecond off the boundary does not fire (no tolerance).
        jitter = datetime.timedelta(microseconds=1)
        assert _tc().is_active(_T0 + 2 * _DT + jitter) is False

    def test_frozen_dataclass(self) -> None:
        tc = _tc()
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.interval = datetime.timedelta(seconds=1)  # type: ignore[misc]

    def test_validate_interval_accepts_integer_multiple(self) -> None:
        _tc(interval=2 * _DT).validate_interval(_DT)

    def test_validate_interval_rejects_non_multiple(self) -> None:
        with pytest.raises(ValueError, match="integer multiple"):
            _tc(interval=1.5 * _DT).validate_interval(_DT)

    def test_validate_interval_rejects_zero_interval_when_enabled(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _tc(interval=datetime.timedelta(0)).validate_interval(_DT)

    def test_validate_interval_skips_disabled_process(self) -> None:
        _tc(interval=1.5 * _DT, enable_process=False).validate_interval(_DT)


def test_physics_process_construction() -> None:
    class _DummyComponent:
        inputs_properties = {}
        outputs_properties = {}

        def __call__(self, state, time_step):
            return {}

    state = RecordingComponentState()
    proc = PhysicsProcess(
        name="muphys",
        component=_DummyComponent(),
        state=state,
        time_control=_tc(),
    )
    assert proc.name == "muphys"
    assert proc.component is not None
    assert proc.state is state
    assert proc.time_control.enable_process


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
class RecordingComponentState(ComponentState):
    """Stub ComponentState: records collect_inputs calls; fixed dict from as_component_input."""

    collect_calls: list = dataclasses.field(default_factory=list)

    def collect_inputs(self, entry_state) -> None:
        self.collect_calls.append(entry_state)

    def as_component_input(self) -> dict:
        return {"foo": "bar"}


@dataclasses.dataclass
class RecordingCoupling:
    """Stub for the whole PhysicsState layer, recording the driver's coupling calls.

    Plays entry state, accumulators, and apply at once — the driver only cares
    about the call sequence, which `events` captures in order.
    """

    events: list = dataclasses.field(default_factory=list)

    # EntryState surface
    def diagnose_from(self, prognostic, tracers) -> None:
        self.events.append(("diagnose", prognostic))

    # TendencyAccumulators surface
    def zero(self) -> None:
        self.events.append(("zero",))

    def accumulate(self, outputs, outputs_properties) -> None:
        self.events.append(("accumulate", dict(outputs)))

    # ApplyToPrognostic surface
    def __call__(self, entry_state, accumulators, dt_seconds) -> None:
        self.events.append(("apply", dt_seconds))


def _driver(processes) -> tuple[PhysicsDriver, RecordingCoupling]:
    coupling = RecordingCoupling()
    driver = PhysicsDriver(
        processes=processes,
        entry_state=coupling,
        accumulators=coupling,
        apply_to_prognostic=coupling,
    )
    return driver, coupling


def test_run_diagnoses_once_accumulates_each_process_and_applies_once() -> None:
    state = RecordingComponentState()
    comp_a = RecordingComponent(
        outputs={"tend_temperature": "A"},
        output_kinds={"tend_temperature": "tendency"},
    )
    comp_b = RecordingComponent(
        outputs={"tend_temperature": "B", "kh": "KH"},
        output_kinds={"tend_temperature": "tendency", "kh": "diagnostic"},
    )
    driver, coupling = _driver(
        [
            PhysicsProcess(name="A", component=comp_a, state=state, time_control=_tc()),
            PhysicsProcess(name="B", component=comp_b, state=state, time_control=_tc()),
        ]
    )

    driver.run(
        prognostic="prog",
        tracers="tracers",
        dtime=_DT,
        simulation_current_datetime=_T0 + _DT,
    )

    assert comp_a.call_count == 1
    assert comp_b.call_count == 1
    # parallel coupling: diagnose + zero once at entry, one accumulate per process,
    # exactly one apply at the very end
    assert coupling.events == [
        ("diagnose", "prog"),
        ("zero",),
        ("accumulate", {"tend_temperature": "A"}),
        ("accumulate", {"tend_temperature": "B", "kh": "KH"}),
        ("apply", 300.0),
    ]
    # both processes were gathered on the same (frozen) entry state
    assert state.collect_calls == [coupling, coupling]
    # non-tendency outputs land in the driver's diagnostics store, by process
    assert driver.diagnostics["B"] == {"kh": "KH"}
    assert driver.diagnostics["A"] == {}


def test_run_raises_for_non_multiple_interval() -> None:
    state = RecordingComponentState()
    comp = RecordingComponent(
        outputs={"tend_temperature": "X"},
        output_kinds={"tend_temperature": "tendency"},
    )
    driver, _ = _driver(
        [
            PhysicsProcess(
                name="X", component=comp, state=state, time_control=_tc(interval=1.5 * _DT)
            ),
        ]
    )

    with pytest.raises(ValueError, match="integer multiple"):
        driver.run(
            prognostic="prog",
            tracers="tracers",
            dtime=_DT,
            simulation_current_datetime=_T0,
        )
    assert comp.call_count == 0


def test_disabled_process_is_never_collected() -> None:
    state = RecordingComponentState()
    comp = RecordingComponent(
        outputs={"tend_temperature": "X"},
        output_kinds={"tend_temperature": "tendency"},
    )
    driver, coupling = _driver(
        [
            PhysicsProcess(
                name="disabled",
                component=comp,
                state=state,
                time_control=_tc(enable_process=False),
            )
        ]
    )

    driver.run(
        prognostic="prog",
        tracers="tracers",
        dtime=_DT,
        simulation_current_datetime=_T0,
    )

    assert comp.call_count == 0
    assert state.collect_calls == []
    # entry diagnosis and the (empty) apply still frame the step
    assert coupling.events == [("diagnose", "prog"), ("zero",), ("apply", 300.0)]


def test_out_of_window_process_does_nothing() -> None:
    state = RecordingComponentState()
    comp = RecordingComponent(
        outputs={"tend_temperature": "X"},
        output_kinds={"tend_temperature": "tendency"},
    )
    # Window starts in the future — the step being integrated is before it.
    future = _T0 + datetime.timedelta(days=1)
    tc = _tc(start=future, end=future + datetime.timedelta(hours=1))
    driver, coupling = _driver(
        [PhysicsProcess(name="future", component=comp, state=state, time_control=tc)]
    )

    driver.run(
        prognostic="prog",
        tracers="tracers",
        dtime=_DT,
        simulation_current_datetime=_T0,
    )

    assert comp.call_count == 0
    assert state.collect_calls == []
    assert coupling.events == [("diagnose", "prog"), ("zero",), ("apply", 300.0)]


def test_inactive_in_window_recycles_cached_outputs() -> None:
    state = RecordingComponentState()
    # Component computes once; on the recycle step it MUST NOT be called, but its
    # cached tendencies accumulate again.
    comp = RecordingComponent(
        outputs={"tend_temperature": "FRESH"},
        output_kinds={"tend_temperature": "tendency"},
    )
    # interval = 2 * dt → process fires every other step.
    driver, coupling = _driver(
        [PhysicsProcess(name="p", component=comp, state=state, time_control=_tc(interval=2 * _DT))]
    )

    # Step 1: active (step start == _T0, elapsed == 0), compute + cache.
    driver.run(
        prognostic="prog", tracers="tracers", dtime=_DT, simulation_current_datetime=_T0 + _DT
    )
    # Step 2: in window, but not active (elapsed == _DT) — recycle the cached outputs.
    driver.run(
        prognostic="prog", tracers="tracers", dtime=_DT, simulation_current_datetime=_T0 + 2 * _DT
    )

    assert comp.call_count == 1
    accumulates = [e for e in coupling.events if e[0] == "accumulate"]
    assert accumulates == [
        ("accumulate", {"tend_temperature": "FRESH"}),
        ("accumulate", {"tend_temperature": "FRESH"}),  # recycled
    ]
    applies = [e for e in coupling.events if e[0] == "apply"]
    assert len(applies) == 2  # one per run


def test_first_in_window_step_inactive_computes_without_keyerror() -> None:
    # Regression (jcanton review): a process whose first-ever in-window step is NOT active
    # (interval = 2*dt, first step lands at start + dt) used to KeyError on the empty recycle
    # cache. With nothing cached to recycle yet, it must compute instead.
    state = RecordingComponentState()
    comp = RecordingComponent(
        outputs={"tend_temperature": "FRESH"},
        output_kinds={"tend_temperature": "tendency"},
    )
    driver, coupling = _driver(
        [PhysicsProcess(name="p", component=comp, state=state, time_control=_tc(interval=2 * _DT))]
    )

    # First call lands in-window but off the firing tick (step start == _T0 + _DT).
    driver.run(
        prognostic="prog", tracers="tracers", dtime=_DT, simulation_current_datetime=_T0 + 2 * _DT
    )

    assert comp.call_count == 1
    assert ("accumulate", {"tend_temperature": "FRESH"}) in coupling.events
