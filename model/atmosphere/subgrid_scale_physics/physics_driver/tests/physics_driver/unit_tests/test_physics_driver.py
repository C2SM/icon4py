# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the physics driver: process loop, time control, sequential coupling, apply-once."""

import dataclasses
import datetime

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.physics_driver import (
    PhysicsProcess,
)
from icon4py.model.common.components import framework as fw, quantities as qty, states
from icon4py.model.common.grid import base as base_grid, simple

from .utils import (
    DT,
    T0,
    RecordingStep,
    driver,
    filled,
    inputs,
    output,
    prognostics,
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


class TemperatureOutput(fw.State):
    tend_temperature: fw.Field[qty.TendencyOfTemperatureOnCellK]


class VerticalWindOutput(fw.State):
    tend_w: fw.Field[qty.TendencyOfWOnCellKHalf]


class TendencyOfPressureOnCellK(fw.Tendency, dims=qty.CELL_K, units="Pa s-1"):
    """A tendency of a physics state leaf the driver does not advance."""


class PressureOutput(fw.State):
    tend_pressure: fw.Field[TendencyOfPressureOnCellK]


@pytest.fixture
def grid() -> base_grid.Grid:
    return simple.simple_grid()


def _qv(tracer_state: states.TracerState) -> np.ndarray:
    assert tracer_state.qv is not None
    return tracer_state.qv.data.asnumpy()


class TestProcessTimeControl:
    def test_is_active_false_when_interval_zero(self) -> None:
        assert time_control(interval=datetime.timedelta(0)).is_active(T0) is False

    def test_is_in_window_at_start_is_true(self) -> None:
        assert time_control().is_in_window(T0) is True

    def test_is_in_window_at_end_is_false(self) -> None:
        end = T0 + datetime.timedelta(hours=1)
        assert time_control(end=end).is_in_window(end) is False

    def test_is_in_window_before_start_is_false(self) -> None:
        assert time_control().is_in_window(T0 - datetime.timedelta(seconds=1)) is False

    def test_is_in_window_inside_is_true(self) -> None:
        assert time_control().is_in_window(T0 + datetime.timedelta(hours=12)) is True

    def test_is_active_at_start_is_true(self) -> None:
        assert time_control().is_active(T0) is True

    def test_is_active_at_one_interval_is_true(self) -> None:
        assert time_control().is_active(T0 + DT) is True

    def test_is_active_at_half_interval_is_false(self) -> None:
        assert time_control().is_active(T0 + DT / 2) is False

    def test_is_active_before_start_is_false(self) -> None:
        assert time_control().is_active(T0 - datetime.timedelta(seconds=1)) is False

    def test_is_active_requires_exact_interval_multiple(self) -> None:
        # Fires only at an exact integer multiple of the interval.
        assert time_control().is_active(T0 + 2 * DT) is True
        # 1 microsecond off the boundary does not fire (no tolerance).
        jitter = datetime.timedelta(microseconds=1)
        assert time_control().is_active(T0 + 2 * DT + jitter) is False

    def test_frozen_dataclass(self) -> None:
        tc = time_control()
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.interval = datetime.timedelta(seconds=1)  # type: ignore[misc]

    def test_validate_interval_accepts_integer_multiple(self) -> None:
        time_control(interval=2 * DT).validate_interval(DT)

    def test_validate_interval_rejects_non_multiple(self) -> None:
        with pytest.raises(ValueError, match="integer multiple"):
            time_control(interval=1.5 * DT).validate_interval(DT)

    def test_validate_interval_rejects_zero_interval(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            time_control(interval=datetime.timedelta(0)).validate_interval(DT)


def test_physics_process_construction(grid: base_grid.Grid) -> None:
    step = RecordingStep(filled(MoistOutput, grid))
    process = PhysicsProcess(name="muphys", step=step, time_control=time_control())
    assert process.name == "muphys"
    assert process.step is step
    assert process.time_control.interval == DT


def test_run_advances_the_physics_state_after_each_process_and_applies_the_sum_once(
    grid: base_grid.Grid,
) -> None:
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)
    assert tracer_state.qv is not None
    step_a = RecordingStep(filled(MoistOutput, grid, tend_qv=1e-7, pflx=5.0))
    step_b = RecordingStep(
        filled(MoistOutput, grid, tend_qv=2e-7, pflx=7.0), watch={"qv": tracer_state.qv}
    )
    step_c = RecordingStep(fw.Empty())
    physics = driver(
        grid,
        [
            PhysicsProcess(name="a", step=step_a, time_control=time_control()),
            PhysicsProcess(name="b", step=step_b, time_control=time_control()),
            PhysicsProcess(name="c", step=step_c, time_control=time_control()),
        ],
    )

    physics.run(inputs(prognostic, tracer_state), out=output(prognostic, tracer_state))

    dt = DT.total_seconds()
    # one physics state, handed to every process
    assert len(step_a.states) == len(step_b.states) == len(step_c.states) == 1
    assert step_b.states[0] is step_a.states[0]
    assert step_c.states[0] is step_a.states[0]
    # sequential coupling: a reads the entry qv, b the qv a advanced, c the qv both advanced
    # (each by its own tendency, not by the running sum)
    np.testing.assert_array_equal(step_a.read[0]["qv"], 1e-3)
    np.testing.assert_allclose(step_b.read[0]["qv"], 1e-3 + dt * 1e-7, rtol=1e-12)
    np.testing.assert_allclose(step_c.read[0]["qv"], 1e-3 + dt * 3e-7, rtol=1e-12)
    # the prognostic qv is untouched while the processes run ...
    np.testing.assert_array_equal(step_b.watched[0]["qv"], 1e-3)
    # ... and gets the sum of the tendencies once; the diagnostics are not accumulated
    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + dt * 3e-7, rtol=1e-12)
    assert set(physics.accumulators) == {"tend_qv"}
    # the driver keeps each process's last output
    assert physics.outputs == {"a": step_a.output, "b": step_b.output, "c": step_c.output}


def test_run_advances_the_temperature_and_applies_the_sum_once_from_the_entry_temperature(
    grid: base_grid.Grid,
) -> None:
    dt = DT.total_seconds()
    step_a = RecordingStep(filled(TemperatureOutput, grid, tend_temperature=1e-3))
    step_b = RecordingStep(fw.Empty())
    physics = driver(
        grid,
        [
            PhysicsProcess(name="a", step=step_a, time_control=time_control()),
            PhysicsProcess(name="b", step=step_b, time_control=time_control()),
        ],
    )
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    physics.run(inputs(prognostic, tracer_state), out=output(prognostic, tracer_state))

    # a reads the diagnosed temperature, b that temperature advanced by a's tendency
    entry_temperature = step_a.read[0]["temperature"]
    np.testing.assert_array_equal(entry_temperature, physics.diagnostics.temperature.data.asnumpy())
    advanced = step_b.read[0]["temperature"]
    np.testing.assert_allclose(advanced - entry_temperature, dt * 1e-3, rtol=1e-12)
    # the final state holds the entry temperature plus the tendency once: diagnosed again,
    # its temperature is the one b read (an update from that advanced temperature would add
    # the tendency twice)
    diagnosis = driver(grid)
    diagnosis.run(inputs(prognostic, tracer_state), out=output(prognostic, tracer_state))
    np.testing.assert_allclose(
        diagnosis.diagnostics.temperature.data.asnumpy(), advanced, rtol=1e-13
    )

    # the same tendency split across two processes gives the same final state
    halves = driver(
        grid,
        [
            PhysicsProcess(
                name=name,
                step=RecordingStep(filled(TemperatureOutput, grid, tend_temperature=0.5e-3)),
                time_control=time_control(),
            )
            for name in ("c", "d")
        ],
    )
    split_prognostic, split_tracers = prognostics(grid), tracers(grid)
    halves.run(inputs(split_prognostic, split_tracers), out=output(split_prognostic, split_tracers))
    for name in ("exner", "theta_v"):
        np.testing.assert_array_equal(
            getattr(split_prognostic, name).data.asnumpy(),
            getattr(prognostic, name).data.asnumpy(),
            err_msg=name,
        )


def test_run_advances_w_for_the_next_process_and_applies_its_tendency_once(
    grid: base_grid.Grid,
) -> None:
    prognostic, tracer_state = prognostics(grid, w=0.5), tracers(grid)
    step_a = RecordingStep(filled(VerticalWindOutput, grid, tend_w=1e-4))
    step_b = RecordingStep(fw.Empty(), watch={"w": prognostic.w})
    physics = driver(
        grid,
        [
            PhysicsProcess(name="a", step=step_a, time_control=time_control()),
            PhysicsProcess(name="b", step=step_b, time_control=time_control()),
        ],
    )

    physics.run(inputs(prognostic, tracer_state), out=output(prognostic, tracer_state))

    dt = DT.total_seconds()
    np.testing.assert_array_equal(step_a.read[0]["w"], 0.5)
    np.testing.assert_allclose(step_b.read[0]["w"], 0.5 + dt * 1e-4, rtol=1e-12)
    np.testing.assert_array_equal(step_b.watched[0]["w"], 0.5)
    np.testing.assert_allclose(prognostic.w.data.asnumpy(), 0.5 + dt * 1e-4, rtol=1e-12)


def test_run_advances_the_cell_wind_for_the_next_process_and_updates_vn_once(
    grid: base_grid.Grid,
) -> None:
    # the neutral geometry of `driver`: zero RBF coefficients, so the diagnosed (u, v) is zero,
    # and the projection of a uniform u-tendency onto the edges is the identity
    prognostic, tracer_state = prognostics(grid), tracers(grid)
    step_a = RecordingStep(filled(WindOutput, grid, tend_u=1e-4))
    step_b = RecordingStep(fw.Empty(), watch={"vn": prognostic.vn})
    physics = driver(
        grid,
        [
            PhysicsProcess(name="a", step=step_a, time_control=time_control()),
            PhysicsProcess(name="b", step=step_b, time_control=time_control()),
        ],
    )

    physics.run(inputs(prognostic, tracer_state), out=output(prognostic, tracer_state))

    dt = DT.total_seconds()
    np.testing.assert_array_equal(step_a.read[0]["u"], 0.0)
    np.testing.assert_allclose(step_b.read[0]["u"], dt * 1e-4, rtol=1e-12)
    np.testing.assert_array_equal(step_b.read[0]["v"], 0.0)
    np.testing.assert_array_equal(step_b.watched[0]["vn"], 0.0)
    np.testing.assert_allclose(prognostic.vn.data.asnumpy(), dt * 1e-4, rtol=1e-12)


def test_run_rejects_a_tendency_of_a_leaf_the_driver_does_not_advance(
    grid: base_grid.Grid,
) -> None:
    step = RecordingStep(filled(PressureOutput, grid, tend_pressure=1.0))
    physics = driver(grid, [PhysicsProcess(name="p", step=step, time_control=time_control())])
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    with pytest.raises(ValueError, match="'tend_pressure'"):
        physics.run(inputs(prognostic, tracer_state), out=output(prognostic, tracer_state))


def test_accumulators_are_zeroed_between_steps(grid: base_grid.Grid) -> None:
    step = RecordingStep(filled(MoistOutput, grid, tend_qv=1e-7))
    physics = driver(grid, [PhysicsProcess(name="p", step=step, time_control=time_control())])
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)

    physics.run(inputs(prognostic, tracer_state), out=output(prognostic, tracer_state))
    physics.run(
        inputs(prognostic, tracer_state, simulation_current_datetime=T0 + 2 * DT),
        out=output(prognostic, tracer_state),
    )

    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + 2 * DT.total_seconds() * 1e-7, rtol=1e-12)


def test_run_raises_for_non_multiple_interval(grid: base_grid.Grid) -> None:
    step = RecordingStep(filled(MoistOutput, grid, tend_qv=1e-7))
    physics = driver(
        grid,
        [PhysicsProcess(name="x", step=step, time_control=time_control(interval=1.5 * DT))],
    )
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    with pytest.raises(ValueError, match="integer multiple"):
        physics.run(
            inputs(prognostic, tracer_state, simulation_current_datetime=T0),
            out=output(prognostic, tracer_state),
        )
    assert step.states == []


def test_out_of_window_process_does_nothing(grid: base_grid.Grid) -> None:
    step = RecordingStep(filled(MoistOutput, grid, tend_qv=1e-7))
    # the window starts in the future: the step being integrated is before it
    future = T0 + datetime.timedelta(days=1)
    window = time_control(start=future, end=future + datetime.timedelta(hours=1))
    physics = driver(grid, [PhysicsProcess(name="future", step=step, time_control=window)])
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)

    physics.run(
        inputs(prognostic, tracer_state, simulation_current_datetime=T0),
        out=output(prognostic, tracer_state),
    )

    assert step.states == []
    assert physics.outputs == {}
    np.testing.assert_array_equal(_qv(tracer_state), 1e-3)


def test_inactive_in_window_recycles_the_last_output(grid: base_grid.Grid) -> None:
    step = RecordingStep(filled(MoistOutput, grid, tend_qv=1e-7))
    # the next process, every step
    after = RecordingStep(fw.Empty())
    # fires every other step
    physics = driver(
        grid,
        [
            PhysicsProcess(name="p", step=step, time_control=time_control(interval=2 * DT)),
            PhysicsProcess(name="after", step=after, time_control=time_control()),
        ],
    )
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)

    # step 1: active (step start == T0), computes
    physics.run(
        inputs(prognostic, tracer_state, simulation_current_datetime=T0 + DT),
        out=output(prognostic, tracer_state),
    )
    # step 2: in the window but not active (step start == T0 + DT), reuses the last output
    physics.run(
        inputs(prognostic, tracer_state, simulation_current_datetime=T0 + 2 * DT),
        out=output(prognostic, tracer_state),
    )

    dt = DT.total_seconds()
    # the physics state reaches the process only on the step it computes
    assert len(step.states) == 1
    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + 2 * dt * 1e-7, rtol=1e-12)
    # the recycled output advances the physics state for the next process too
    np.testing.assert_allclose(after.read[1]["qv"], 1e-3 + 2 * dt * 1e-7, rtol=1e-12)


def test_first_in_window_step_inactive_computes(grid: base_grid.Grid) -> None:
    # A process whose first step in its window is not active (interval = 2 dt, the first
    # step starts at T0 + DT) has no output to reuse yet: it computes instead.
    step = RecordingStep(filled(MoistOutput, grid, tend_qv=1e-7))
    physics = driver(
        grid,
        [PhysicsProcess(name="p", step=step, time_control=time_control(interval=2 * DT))],
    )
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)

    physics.run(
        inputs(prognostic, tracer_state, simulation_current_datetime=T0 + 2 * DT),
        out=output(prognostic, tracer_state),
    )

    assert len(step.states) == 1
    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + DT.total_seconds() * 1e-7, rtol=1e-12)


def test_run_requires_every_moisture_species(grid: base_grid.Grid) -> None:
    physics = driver(grid)
    prognostic, tracer_state = prognostics(grid), tracers(grid)
    without_qc = dataclasses.replace(tracer_state, qc=None)

    with pytest.raises(ValueError, match="qc"):
        physics.run(inputs(prognostic, without_qc), out=output(prognostic, without_qc))


def test_out_need_not_alias_the_inputs(grid: base_grid.Grid) -> None:
    step = RecordingStep(filled(MoistOutput, grid, tend_qv=1e-7))
    physics = driver(grid, [PhysicsProcess(name="p", step=step, time_control=time_control())])
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)
    new_prognostic, new_tracers = prognostics(grid, theta_v=0.0), tracers(grid, qv=0.0)

    physics.run(inputs(prognostic, tracer_state), out=output(new_prognostic, new_tracers))

    # the inputs are read only; out continues them, with the tendencies applied
    np.testing.assert_array_equal(_qv(tracer_state), 1e-3)
    np.testing.assert_allclose(_qv(new_tracers), 1e-3 + DT.total_seconds() * 1e-7, rtol=1e-12)
    np.testing.assert_array_equal(new_prognostic.theta_v.data.asnumpy(), 300.0)


def test_apply_updates_tracers_w_and_thermodynamics_once(grid: base_grid.Grid) -> None:
    step = RecordingStep(
        filled(ThermoOutput, grid, tend_qv=1e-7, tend_temperature=1e-3, tend_w=1e-4)
    )
    physics = driver(grid, [PhysicsProcess(name="p", step=step, time_control=time_control())])
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)
    exner_before = prognostic.exner.data.asnumpy().copy()
    theta_v_before = prognostic.theta_v.data.asnumpy().copy()

    physics.run(inputs(prognostic, tracer_state), out=output(prognostic, tracer_state))

    dt = DT.total_seconds()
    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + 1e-7 * dt, rtol=1e-12)
    np.testing.assert_allclose(prognostic.w.data.asnumpy(), 1e-4 * dt, rtol=1e-12)
    # EOS wiring smoke test: the exact-EOS update rewrote exner and theta_v (no direction
    # assertion: the uniform test state is not EOS-consistent)
    assert not np.array_equal(prognostic.exner.data.asnumpy(), exner_before)
    assert not np.array_equal(prognostic.theta_v.data.asnumpy(), theta_v_before)


def test_apply_projects_the_wind_tendencies_onto_vn(grid: base_grid.Grid) -> None:
    # uniform tend_u = 1e-4, tend_v = 0 with the neutral geometry of `driver`:
    # ddt_vn = 2 * 0.5 * 1e-4 * 1.0 = 1e-4 on every edge of the periodic simple grid
    step = RecordingStep(filled(WindOutput, grid, tend_u=1e-4))
    physics = driver(grid, [PhysicsProcess(name="p", step=step, time_control=time_control())])
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    physics.run(inputs(prognostic, tracer_state), out=output(prognostic, tracer_state))

    np.testing.assert_allclose(prognostic.vn.data.asnumpy(), 1e-4 * DT.total_seconds(), rtol=1e-12)


def test_apply_rejects_a_lone_horizontal_wind_tendency(grid: base_grid.Grid) -> None:
    # vn is one projection of (u, v): a process emitting only one of the two would
    # silently lose the other half of the momentum, an error rather than a no-op
    step = RecordingStep(filled(LoneWindOutput, grid, tend_u=1e-4))
    physics = driver(grid, [PhysicsProcess(name="p", step=step, time_control=time_control())])
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    with pytest.raises(ValueError, match="applied as a pair"):
        physics.run(inputs(prognostic, tracer_state), out=output(prognostic, tracer_state))
