# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the physics driver: process loop, time control, accumulation, apply-once."""

import dataclasses
import datetime

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import (
    physics_driver,
    physics_state,
)
from icon4py.model.common import framework as fw
from icon4py.model.common.framework import quantities as qty, states
from icon4py.model.common.grid import base as base_grid, simple

from .utils import (
    DT,
    T0,
    Recorder,
    driver,
    filled,
    inputs,
    output,
    prognostics,
    recording,
    time_control,
    tracers,
)


class MoistOutput(fw.State):
    """A process output with a tendency and a diagnostic."""

    tend_qv: fw.Field[qty.TendencyOfQvOnCellK]
    pflx: fw.Field[qty.PrecipitationFluxOnCellK]


class ThermoOutput(fw.State):
    tend_qv: fw.Field[qty.TendencyOfQvOnCellK]
    tend_temperature: fw.Field[qty.TendencyOfTemperatureOnCellK]
    tend_w: fw.Field[qty.TendencyOfWOnCellKHalf]


class WindOutput(fw.State):
    tend_u: fw.Field[qty.TendencyOfUOnCellK]
    tend_v: fw.Field[qty.TendencyOfVOnCellK]


class LoneWindOutput(fw.State):
    tend_u: fw.Field[qty.TendencyOfUOnCellK]


@pytest.fixture
def grid() -> base_grid.Grid:
    return simple.simple_grid()


def _qv(tracer_state: states.TracerState) -> np.ndarray:
    assert tracer_state.qv is not None
    return tracer_state.qv.asnumpy()


class TestProcessTimeControl:
    @pytest.mark.parametrize("interval", [datetime.timedelta(0), -DT])
    def test_rejects_a_non_positive_interval(self, interval: datetime.timedelta) -> None:
        with pytest.raises(ValueError, match="positive"):
            time_control(interval=interval)

    def test_is_in_window_at_start_is_true(self) -> None:
        assert time_control().is_in_window(T0) is True

    def test_is_in_window_at_end_is_false(self) -> None:
        end = T0 + datetime.timedelta(hours=1)
        assert time_control(end=end).is_in_window(end) is False

    def test_is_in_window_before_start_is_false(self) -> None:
        assert time_control().is_in_window(T0 - datetime.timedelta(seconds=1)) is False

    def test_is_in_window_inside_is_true(self) -> None:
        assert time_control().is_in_window(T0 + datetime.timedelta(hours=12)) is True

    def test_computes_at_start(self) -> None:
        assert time_control().computes_at(T0) is True

    def test_computes_at_one_interval(self) -> None:
        assert time_control().computes_at(T0 + DT) is True

    def test_does_not_compute_at_half_interval(self) -> None:
        assert time_control().computes_at(T0 + DT / 2) is False

    def test_does_not_compute_before_start(self) -> None:
        assert time_control().computes_at(T0 - datetime.timedelta(seconds=1)) is False

    def test_computes_at_exact_interval_multiples_only(self) -> None:
        assert time_control().computes_at(T0 + 2 * DT) is True
        # 1 microsecond off the boundary does not compute (no tolerance).
        jitter = datetime.timedelta(microseconds=1)
        assert time_control().computes_at(T0 + 2 * DT + jitter) is False

    def test_frozen_dataclass(self) -> None:
        tc = time_control()
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.interval = datetime.timedelta(seconds=1)  # type: ignore[misc]


def test_step_computes_when_active_reuses_otherwise_and_skips_outside_the_window(
    grid: base_grid.Grid,
) -> None:
    entry = fw.allocate(physics_state.EntryState, grid, None)
    window_end = T0 + 4 * DT
    outputs = filled(MoistOutput, grid)
    process, recorder = recording(outputs, time_control(interval=2 * DT, end=window_end))

    # the first step in the window computes even when it is not active
    assert process.step(entry, T0 + DT) is outputs
    # active: computes again
    assert process.step(entry, T0 + 2 * DT) is outputs
    # inactive: reuses the last output without collecting an input
    assert process.step(entry, T0 + 3 * DT) is outputs
    assert len(recorder.entries) == 2
    # outside the window: nothing
    assert process.step(entry, window_end) is None
    assert len(recorder.entries) == 2


def test_run_hands_every_process_the_same_entry_state_and_applies_the_sum_once(
    grid: base_grid.Grid,
) -> None:
    process_a, recorder_a = recording(
        filled(MoistOutput, grid, tend_qv=1e-7, pflx=5.0), time_control()
    )
    process_b, recorder_b = recording(
        filled(MoistOutput, grid, tend_qv=2e-7, pflx=7.0), time_control()
    )
    physics = driver(grid, [process_a, process_b])
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)

    physics.run(inputs(prognostic, tracer_state), outputs=output(prognostic, tracer_state))

    # parallel coupling: one entry state, diagnosed once, read by both processes
    assert len(recorder_a.entries) == 1
    assert recorder_b.entries == recorder_a.entries
    assert recorder_b.entries[0] is recorder_a.entries[0]
    # b reads the entry qv, not qv already updated by a's tendency
    np.testing.assert_array_equal(recorder_a.qv_read[0], 1e-3)
    np.testing.assert_array_equal(recorder_b.qv_read[0], 1e-3)
    # the tendencies are summed and applied once; the diagnostics are not accumulated
    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + DT.total_seconds() * 3e-7, rtol=1e-12)
    assert set(physics.accumulators) == {"tend_qv"}


def test_accumulators_are_zeroed_between_steps(grid: base_grid.Grid) -> None:
    process, _ = recording(filled(MoistOutput, grid, tend_qv=1e-7), time_control())
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)

    physics.run(inputs(prognostic, tracer_state), outputs=output(prognostic, tracer_state))
    physics.run(
        inputs(prognostic, tracer_state, simulation_current_datetime=T0 + 2 * DT),
        outputs=output(prognostic, tracer_state),
    )

    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + 2 * DT.total_seconds() * 1e-7, rtol=1e-12)


def test_out_of_window_process_does_nothing(grid: base_grid.Grid) -> None:
    # the window starts in the future: the step being integrated is before it
    future = T0 + datetime.timedelta(days=1)
    window = time_control(start=future, end=future + datetime.timedelta(hours=1))
    process, recorder = recording(filled(MoistOutput, grid, tend_qv=1e-7), window)
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)

    physics.run(
        inputs(prognostic, tracer_state, simulation_current_datetime=T0),
        outputs=output(prognostic, tracer_state),
    )

    assert recorder.entries == []
    np.testing.assert_array_equal(_qv(tracer_state), 1e-3)


def test_inactive_in_window_recycles_the_last_output(grid: base_grid.Grid) -> None:
    process, recorder = recording(
        filled(MoistOutput, grid, tend_qv=1e-7), time_control(interval=2 * DT)
    )
    # fires every other step
    physics = driver(
        grid,
        [process],
    )
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)

    # step 1: active (step start == T0), computes
    physics.run(
        inputs(prognostic, tracer_state, simulation_current_datetime=T0 + DT),
        outputs=output(prognostic, tracer_state),
    )
    # step 2: in the window but not active (step start == T0 + DT), reuses the last output
    physics.run(
        inputs(prognostic, tracer_state, simulation_current_datetime=T0 + 2 * DT),
        outputs=output(prognostic, tracer_state),
    )

    # the entry state reaches the process only on the step it computes
    assert len(recorder.entries) == 1
    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + 2 * DT.total_seconds() * 1e-7, rtol=1e-12)


def test_first_in_window_step_inactive_computes(grid: base_grid.Grid) -> None:
    # A process whose first step in its window is not active (interval = 2 dt, the first
    # step starts at T0 + DT) has no output to reuse yet: it computes instead.
    process, recorder = recording(
        filled(MoistOutput, grid, tend_qv=1e-7), time_control(interval=2 * DT)
    )
    physics = driver(
        grid,
        [process],
    )
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)

    physics.run(
        inputs(prognostic, tracer_state, simulation_current_datetime=T0 + 2 * DT),
        outputs=output(prognostic, tracer_state),
    )

    assert len(recorder.entries) == 1
    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + DT.total_seconds() * 1e-7, rtol=1e-12)


def test_run_requires_every_moisture_species(grid: base_grid.Grid) -> None:
    physics = driver(grid)
    prognostic, tracer_state = prognostics(grid), tracers(grid)
    without_qc = dataclasses.replace(tracer_state, qc=None)

    with pytest.raises(ValueError, match="qc"):
        physics.run(inputs(prognostic, without_qc), outputs=output(prognostic, without_qc))


def test_out_need_not_alias_the_inputs(grid: base_grid.Grid) -> None:
    process, _ = recording(filled(MoistOutput, grid, tend_qv=1e-7), time_control())
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)
    new_prognostic, new_tracers = prognostics(grid, theta_v=0.0), tracers(grid, qv=0.0)

    physics.run(inputs(prognostic, tracer_state), outputs=output(new_prognostic, new_tracers))

    # the inputs are read only; out continues them, with the tendencies applied
    np.testing.assert_array_equal(_qv(tracer_state), 1e-3)
    np.testing.assert_allclose(_qv(new_tracers), 1e-3 + DT.total_seconds() * 1e-7, rtol=1e-12)
    np.testing.assert_array_equal(new_prognostic.theta_v.asnumpy(), 300.0)


def test_apply_updates_tracers_w_and_thermodynamics_once(grid: base_grid.Grid) -> None:
    process, _ = recording(
        filled(ThermoOutput, grid, tend_qv=1e-7, tend_temperature=1e-3, tend_w=1e-4),
        time_control(),
    )
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)
    exner_before = prognostic.exner.asnumpy().copy()
    theta_v_before = prognostic.theta_v.asnumpy().copy()

    physics.run(inputs(prognostic, tracer_state), outputs=output(prognostic, tracer_state))

    dt = DT.total_seconds()
    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + 1e-7 * dt, rtol=1e-12)
    np.testing.assert_allclose(prognostic.w.asnumpy(), 1e-4 * dt, rtol=1e-12)
    # EOS wiring smoke test: the exact-EOS update rewrote exner and theta_v (no direction
    # assertion: the uniform test state is not EOS-consistent)
    assert not np.array_equal(prognostic.exner.asnumpy(), exner_before)
    assert not np.array_equal(prognostic.theta_v.asnumpy(), theta_v_before)


def test_apply_projects_the_wind_tendencies_onto_vn(grid: base_grid.Grid) -> None:
    # uniform tend_u = 1e-4, tend_v = 0 with the neutral geometry of `driver`:
    # ddt_vn = 2 * 0.5 * 1e-4 * 1.0 = 1e-4 on every edge of the periodic simple grid
    process, _ = recording(filled(WindOutput, grid, tend_u=1e-4), time_control())
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    physics.run(inputs(prognostic, tracer_state), outputs=output(prognostic, tracer_state))

    np.testing.assert_allclose(prognostic.vn.asnumpy(), 1e-4 * DT.total_seconds(), rtol=1e-12)


def test_apply_rejects_a_lone_horizontal_wind_tendency(grid: base_grid.Grid) -> None:
    # vn is one projection of (u, v): a process emitting only one of the two would
    # silently lose the other half of the momentum, an error rather than a no-op
    process, _ = recording(filled(LoneWindOutput, grid, tend_u=1e-4), time_control())
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    with pytest.raises(ValueError, match="applied as a pair"):
        physics.run(inputs(prognostic, tracer_state), outputs=output(prognostic, tracer_state))


def _empty(entry: physics_state.EntryState) -> fw.Empty:
    return fw.Empty()


def static_checks(outputs: MoistOutput) -> None:
    """
    A wiring error mypy reports; never run. With `warn_unused_ignores`, the check fails when
    it stops being an error.
    """
    # a collect_input that builds another component's Input
    physics_driver.PhysicsProcess(
        component=Recorder(),
        collect_input=_empty,  # type: ignore[arg-type]
        time_control=time_control(),
        outputs=outputs,
    )
