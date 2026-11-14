# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the physics driver: dyn2phy, process loop, time control, sequential coupling, phy2dyn."""

import dataclasses
import datetime
from typing import Any

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import (
    physics_process,
    physics_state,
)
from icon4py.model.common import dimension as dims, field, framework as fw
from icon4py.model.common.framework import quantities as qty, states
from icon4py.model.common.grid import base as base_grid, simple

from .utils import (
    DT,
    T0,
    Recorder,
    RecordingExchange,
    driver,
    filled,
    make_input,
    make_output,
    prognostics,
    recording,
    time_control,
    tracers,
    zero_physics_state,
)


class MoistOutput(fw.State):
    """A process output with a tendency and a diagnostic."""

    tend_qv: field.Field[qty.TendencyOfQvOnCellK]
    pflx: field.Field[qty.PrecipitationFluxOnCellK]


class WindOutput(fw.State):
    tend_u: field.Field[qty.TendencyOfUOnCellK]
    tend_v: field.Field[qty.TendencyOfVOnCellK]


class LoneWindOutput(fw.State):
    tend_u: field.Field[qty.TendencyOfUOnCellK]


@pytest.fixture
def grid() -> base_grid.Grid:
    return simple.simple_grid()


def _qv(tracer_state: states.TracerState) -> np.ndarray:
    assert tracer_state.qv is not None
    return tracer_state.qv.asnumpy()


class TestProcessTimeControl:
    @pytest.mark.parametrize("process_dtime", [datetime.timedelta(0), -DT])
    def test_rejects_a_non_positive_process_dtime(self, process_dtime: datetime.timedelta) -> None:
        with pytest.raises(ValueError, match="positive"):
            time_control(process_dtime=process_dtime)

    def test_rejects_a_process_dtime_not_a_multiple_of_driver_dtime(self) -> None:
        with pytest.raises(ValueError, match="positive multiple"):
            time_control(process_dtime=DT * 3 / 2)

    @pytest.mark.parametrize("offset", [DT / 2, -DT / 2])
    def test_rejects_a_window_start_between_two_driver_steps(
        self, offset: datetime.timedelta
    ) -> None:
        with pytest.raises(ValueError, match="window_start"):
            time_control(start=T0 + offset)

    def test_accepts_a_window_start_on_a_driver_step_before_the_simulation_start(self) -> None:
        tc = time_control(start=T0 - 3 * DT, process_dtime=2 * DT)
        # the process grid is anchored at `window_start`, not at the simulation start
        assert tc.computes_at(T0) is False
        assert tc.computes_at(T0 + DT) is True

    def test_skips_steps_only_when_process_dtime_exceeds_driver_dtime(self) -> None:
        assert time_control().skips_steps is False
        assert time_control(process_dtime=2 * DT).skips_steps is True

    @pytest.mark.parametrize(
        "step_start, expected",
        [
            (T0, True),
            (T0 + datetime.timedelta(minutes=30), True),
            (T0 - datetime.timedelta(seconds=1), False),
            (T0 + datetime.timedelta(hours=1), False),
        ],
    )
    def test_is_in_window(self, step_start: datetime.datetime, expected: bool) -> None:
        tc = time_control(end=T0 + datetime.timedelta(hours=1))
        assert tc.is_in_window(step_start) is expected

    @pytest.mark.parametrize(
        "step_start, expected",
        [
            (T0, True),
            (T0 + DT, True),
            (T0 + 2 * DT, True),
            (T0 + DT / 2, False),
            (T0 - datetime.timedelta(seconds=1), False),
            # no tolerance
            (T0 + 2 * DT + datetime.timedelta(microseconds=1), False),
        ],
    )
    def test_computes_at(self, step_start: datetime.datetime, expected: bool) -> None:
        assert time_control().computes_at(step_start) is expected


def test_run_zeroes_the_tendency_leaves_of_the_outputs_outside_the_window(
    grid: base_grid.Grid,
) -> None:
    window_end = T0 + 4 * DT
    process, recorder = recording(
        MoistOutput, {}, time_control(process_dtime=2 * DT, end=window_end), grid
    )
    # the output handed to this step, with a tendency and a flux already in them
    given = filled(MoistOutput, grid, tend_qv=1e-7, pflx=5.0)
    state = zero_physics_state(grid, step_start=window_end)

    result = process.run(state, given)

    # the zeroing happens in place: the output handed in is the output handed back
    assert result is given
    # the tendency leaves are zeroed, the non-tendency leaves (the muphys fluxes) are not
    np.testing.assert_array_equal(given.tend_qv.asnumpy(), 0.0)
    np.testing.assert_array_equal(given.pflx.asnumpy(), 5.0)
    # outside its window the process never runs its component
    assert recorder.states == []


def test_run_computes_on_the_first_in_window_step_even_when_not_triggered(
    grid: base_grid.Grid,
) -> None:
    # the process's first step in its window is not one its time control fires; it computes
    # anyway, because there is nothing cached yet
    output = filled(MoistOutput, grid)
    process, recorder = recording(MoistOutput, {}, time_control(process_dtime=2 * DT), grid)
    state = zero_physics_state(grid, step_start=T0 + DT)

    assert process.run(state, output) is output
    assert len(recorder.states) == 1

    # the next step fires the time control again, over the cached values: it computes again
    again = zero_physics_state(grid, step_start=T0 + 2 * DT)

    assert process.run(again, output) is output
    assert len(recorder.states) == 2


def test_run_hands_the_cached_tendencies_to_the_given_output_on_a_non_computing_step(
    grid: base_grid.Grid,
) -> None:
    process, recorder = recording(
        MoistOutput, {"tend_qv": 2e-7, "pflx": 3.0}, time_control(process_dtime=2 * DT), grid
    )
    process.run(zero_physics_state(grid, step_start=T0), filled(MoistOutput, grid))
    # another buffer, with other values in it
    given = filled(MoistOutput, grid, tend_qv=1e-7, pflx=5.0)

    assert process.run(zero_physics_state(grid, step_start=T0 + DT), given) is given

    # the cached tendencies, the other leaves as the caller left them
    np.testing.assert_array_equal(given.tend_qv.asnumpy(), 2e-7)
    np.testing.assert_array_equal(given.pflx.asnumpy(), 5.0)
    # the cache is a copy: what the caller writes into the buffer does not reach it
    given.tend_qv.ndarray[...] = 42.0  # type: ignore[index]  # NDArrayObject declares no __setitem__
    process.run(zero_physics_state(grid, step_start=T0 + 3 * DT), given)
    np.testing.assert_array_equal(given.tend_qv.asnumpy(), 2e-7)
    assert len(recorder.states) == 1


def test_run_advances_the_physics_state_after_each_process(grid: base_grid.Grid) -> None:
    process_a, recorder_a = recording(
        MoistOutput, {"tend_qv": 1e-7, "pflx": 5.0}, time_control(), grid
    )
    process_b, recorder_b = recording(
        MoistOutput, {"tend_qv": 2e-7, "pflx": 7.0}, time_control(), grid
    )
    physics = driver(grid, [process_a, process_b])
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)

    physics.run(make_input(prognostic, tracer_state), output=make_output(prognostic, tracer_state))

    # sequential coupling: one physics state, diagnosed once and advanced in place
    assert len(recorder_a.states) == 1
    assert recorder_b.states[0] is recorder_a.states[0]
    # b reads qv already advanced by a's tendency
    dt = DT.total_seconds()
    np.testing.assert_array_equal(recorder_a.qv_read[0], 1e-3)
    np.testing.assert_allclose(recorder_b.qv_read[0], 1e-3 + dt * 1e-7, rtol=1e-12)
    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + dt * 3e-7, rtol=1e-12)
    # the tendencies are summed over the processes, the diagnostics are not
    assert set(physics.accumulators) == {qty.TendencyOfQvOnCellK}
    np.testing.assert_allclose(
        physics.accumulators[qty.TendencyOfQvOnCellK].asnumpy(), 3e-7, rtol=1e-12
    )


def test_driver_rejects_a_process_timed_for_another_dtime(grid: base_grid.Grid) -> None:
    process, _ = recording(
        MoistOutput, {}, time_control(process_dtime=4 * DT, driver_dtime=2 * DT), grid
    )
    with pytest.raises(ValueError, match="driver's dtime"):
        driver(grid, [process])


def test_accumulators_are_zeroed_between_steps(grid: base_grid.Grid) -> None:
    process, _ = recording(MoistOutput, {"tend_qv": 1e-7}, time_control(), grid)
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)

    physics.run(make_input(prognostic, tracer_state), output=make_output(prognostic, tracer_state))
    physics.run(
        make_input(prognostic, tracer_state, simulation_current_datetime=T0 + 2 * DT),
        output=make_output(prognostic, tracer_state),
    )

    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + 2 * DT.total_seconds() * 1e-7, rtol=1e-12)


def test_inactive_in_window_recycles_the_last_output(grid: base_grid.Grid) -> None:
    process, recorder = recording(
        MoistOutput, {"tend_qv": 1e-7}, time_control(process_dtime=2 * DT), grid
    )
    # fires every other step
    physics = driver(
        grid,
        [process],
    )
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)

    # step 1: active (step start == T0), computes
    physics.run(
        make_input(prognostic, tracer_state, simulation_current_datetime=T0 + DT),
        output=make_output(prognostic, tracer_state),
    )
    # step 2: in the window but not active (step start == T0 + DT), reuses the last output
    physics.run(
        make_input(prognostic, tracer_state, simulation_current_datetime=T0 + 2 * DT),
        output=make_output(prognostic, tracer_state),
    )

    # the physics state reaches the process only on the step it computes
    assert len(recorder.states) == 1
    np.testing.assert_allclose(_qv(tracer_state), 1e-3 + 2 * DT.total_seconds() * 1e-7, rtol=1e-12)


def test_run_requires_every_moisture_species(grid: base_grid.Grid) -> None:
    physics = driver(grid)
    prognostic, tracer_state = prognostics(grid), tracers(grid)
    without_qc = dataclasses.replace(tracer_state, qc=None)

    with pytest.raises(ValueError, match="qc"):
        physics.run(make_input(prognostic, without_qc), output=make_output(prognostic, without_qc))


def test_out_need_not_alias_the_inputs(grid: base_grid.Grid) -> None:
    process, _ = recording(MoistOutput, {"tend_qv": 1e-7}, time_control(), grid)
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid, qv=1e-3)
    new_prognostic, new_tracers = prognostics(grid, theta_v=0.0), tracers(grid, qv=0.0)

    physics.run(
        make_input(prognostic, tracer_state), output=make_output(new_prognostic, new_tracers)
    )

    # The input is read only, `output` continues it: as the same run done in place
    in_place_prognostic, in_place_tracers = prognostics(grid), tracers(grid, qv=1e-3)
    driver(grid, [process]).run(
        make_input(in_place_prognostic, in_place_tracers),
        output=make_output(in_place_prognostic, in_place_tracers),
    )
    np.testing.assert_array_equal(_qv(tracer_state), 1e-3)
    np.testing.assert_array_equal(prognostic.theta_v.asnumpy(), 300.0)
    np.testing.assert_allclose(_qv(new_tracers), 1e-3 + DT.total_seconds() * 1e-7, rtol=1e-12)
    for name, field_ in fw.fields(new_prognostic).items():
        np.testing.assert_array_equal(
            field_.asnumpy(), getattr(in_place_prognostic, name).asnumpy(), err_msg=name
        )


def test_run_projects_the_wind_tendencies_onto_vn(grid: base_grid.Grid) -> None:
    # uniform tend_u = 1e-4, tend_v = 0 with the neutral geometry of `driver`:
    # ddt_vn = 2 * 0.5 * 1e-4 * 1.0 = 1e-4 on every edge of the periodic simple grid
    process, _ = recording(WindOutput, {"tend_u": 1e-4}, time_control(), grid)
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    physics.run(make_input(prognostic, tracer_state), output=make_output(prognostic, tracer_state))

    np.testing.assert_allclose(prognostic.vn.asnumpy(), 1e-4 * DT.total_seconds(), rtol=1e-12)


def test_run_rejects_a_lone_horizontal_wind_tendency(grid: base_grid.Grid) -> None:
    # vn is one projection of (u, v): a process emitting only one of the two would
    # silently lose the other half of the momentum, an error rather than a no-op
    process, _ = recording(LoneWindOutput, {"tend_u": 1e-4}, time_control(), grid)
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    with pytest.raises(ValueError, match="applied as a pair"):
        physics.run(
            make_input(prognostic, tracer_state), output=make_output(prognostic, tracer_state)
        )


class TemperatureOutput(fw.State):
    tend_temperature: field.Field[qty.TendencyOfTemperatureOnCellK]


class VnTendencyOutput(fw.State):
    tend_vn: field.Field[qty.TendencyOfVnOnEdgeK]


class FluxOutput(fw.State):
    sensible_heat_flux: field.Field[qty.SensibleHeatFluxOnCell]
    tend_qv: field.Field[qty.TendencyOfQvOnCellK]


def test_diagnose_views_the_inputs_and_the_driver_diagnostics(grid: base_grid.Grid) -> None:
    process, recorder = recording(fw.Empty, {}, time_control(), grid)
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid)
    air_mass = fw.zeros(qty.AirMassOnCellK, grid, None)
    driver_input = make_input(prognostic, tracer_state, air_mass=air_mass)
    diagnostics = fw.allocate(states.Diagnostics, grid, None)

    physics.run(driver_input, output=make_output(prognostic, tracer_state, diagnostics=diagnostics))

    (state,) = recorder.states
    # no copies: the prognostics and tracers as given, the diagnostics in the driver's buffers
    for name in ("vn", "w", "exner", "theta_v", "rho", "air_mass", *states.TRACERS):
        assert getattr(state, name) is getattr(driver_input, name), name
    for name, field_ in fw.fields(diagnostics).items():
        assert getattr(state, name) is field_, name
    # without a surface process the fluxes are absent
    assert state.sensible_heat_flux is None


def test_run_rejects_a_tendency_of_no_advanced_field(grid: base_grid.Grid) -> None:
    process, _ = recording(VnTendencyOutput, {"tend_vn": 1e-4}, time_control(), grid)
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    with pytest.raises(ValueError, match="'tend_vn' is not the tendency of an advanced field"):
        physics.run(
            make_input(prognostic, tracer_state), output=make_output(prognostic, tracer_state)
        )


def test_run_syncs_the_halos_at_entry_the_wind_sums_and_what_it_wrote(
    grid: base_grid.Grid,
) -> None:
    exchange = RecordingExchange()
    thermo, recorder = recording(
        TemperatureOutput, {"tend_temperature": 1e-3}, time_control(), grid
    )
    wind, _ = recording(WindOutput, {"tend_u": 1e-4}, time_control(), grid)
    physics = driver(grid, [thermo, wind], exchange=exchange)
    prognostic, tracer_state = prognostics(grid), tracers(grid)
    perturbed_exner = fw.zeros(qty.PerturbedExnerOnCellK, grid, None)

    physics.run(
        make_input(prognostic, tracer_state),
        output=make_output(prognostic, tracer_state, perturbed_exner=perturbed_exner),
    )

    # identity, not equality: the syncs must target the very buffers that were read and written,
    # and the wind sums before they are projected onto vn
    (state,) = recorder.states
    sums = physics.accumulators
    assert exchange.events == [
        (dims.CellDim, [id(prognostic.rho), id(state.u), id(state.v), id(prognostic.w)]),
        (dims.CellDim, [id(sums[qty.TendencyOfUOnCellK]), id(sums[qty.TendencyOfVOnCellK])]),
        (
            dims.CellDim,
            [
                *(id(getattr(tracer_state, name)) for name in states.TRACERS),
                id(prognostic.w),
                id(prognostic.exner),
                id(prognostic.theta_v),
                id(perturbed_exner),
            ],
        ),
        (dims.EdgeDim, [id(prognostic.vn)]),
    ]


def test_a_process_output_is_offered_to_the_others_through_the_physics_state(
    grid: base_grid.Grid,
) -> None:
    provider, _ = recording(FluxOutput, {"sensible_heat_flux": 5.0}, time_control(), grid)
    reader, recorder = recording(fw.Empty, {}, time_control(), grid)
    physics = driver(grid, [provider, reader])
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    physics.run(make_input(prognostic, tracer_state), output=make_output(prognostic, tracer_state))

    (state,) = recorder.states
    assert state.sensible_heat_flux is physics._process_outputs[provider].sensible_heat_flux
    np.testing.assert_array_equal(state.sensible_heat_flux.asnumpy(), 5.0)


def test_a_second_provider_of_a_physics_state_leaf_is_rejected(grid: base_grid.Grid) -> None:
    first, _ = recording(FluxOutput, {}, time_control(), grid)
    second, _ = recording(FluxOutput, {}, time_control(), grid)
    with pytest.raises(ValueError, match="'sensible_heat_flux' is computed by more than one"):
        driver(grid, [first, second])


def test_diagnostic_fields_are_keyed_by_standard_name_and_skip_the_tendencies(
    grid: base_grid.Grid,
) -> None:
    process, _ = recording(FluxOutput, {}, time_control(), grid)
    physics = driver(grid, [process])

    ((name, field_),) = physics.diagnostic_fields().items()

    assert name == qty.SensibleHeatFluxOnCell.standard_name
    assert field_ is physics._process_outputs[process].sensible_heat_flux


def _empty(state: physics_state.PhysicsState) -> fw.Empty:
    return fw.Empty()


def static_checks() -> None:
    """
    A wiring error mypy reports; never run. With `warn_unused_ignores`, the check fails when
    it stops being an error.
    """
    # a collect_input that builds another component's Input
    physics_process.PhysicsProcess(
        component=Recorder({}),
        collect_input=_empty,  # type: ignore[arg-type]
        time_control=time_control(),
        grid=simple.simple_grid(),
        allocator=None,
    )
