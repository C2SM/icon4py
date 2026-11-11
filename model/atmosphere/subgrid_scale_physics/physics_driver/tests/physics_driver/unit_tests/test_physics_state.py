# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the physics state the physics driver diagnoses, hands to its processes and advances."""

import numpy as np
import pytest

from icon4py.model.common import dimension as dims, field, framework as fw
from icon4py.model.common.framework import quantities as qty, states
from icon4py.model.common.grid import base as base_grid, simple
from icon4py.model.common.utils import data_allocation as data_alloc

from .utils import (
    RecordingExchange,
    driver,
    make_input,
    make_output,
    prognostics,
    recording,
    time_control,
    tracers,
)


class ThermoOutput(fw.State):
    tend_temperature: field.Field[qty.TendencyOfTemperatureOnCellK]


class WindOutput(fw.State):
    tend_u: field.Field[qty.TendencyOfUOnCellK]
    tend_v: field.Field[qty.TendencyOfVOnCellK]


class VnTendencyOutput(fw.State):
    tend_vn: field.Field[qty.TendencyOfVnOnEdgeK]


class FluxOutput(fw.State):
    sensible_heat_flux: field.Field[qty.SensibleHeatFluxOnCell]
    tend_qv: field.Field[qty.TendencyOfQvOnCellK]


@pytest.fixture
def grid() -> base_grid.Grid:
    return simple.simple_grid()


def test_physics_state_views_the_inputs_and_the_driver_diagnostics(grid: base_grid.Grid) -> None:
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


def test_diagnose_fills_the_diagnostics_and_leaves_the_input_untouched(
    grid: base_grid.Grid,
) -> None:
    """The diagnosis fills plausible fields; `input` is read only, `output` continues it."""
    process, _ = recording(fw.Empty, {}, time_control(), grid)
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid, exner=0.95, theta_v=300.0), tracers(grid)
    before = {name: field_.asnumpy().copy() for name, field_ in fw.fields(prognostic).items()}
    diagnostics = fw.allocate(states.Diagnostics, grid, None)

    physics.run(
        make_input(prognostic, tracer_state),
        output=make_output(prognostics(grid), tracers(grid, qv=0.0), diagnostics=diagnostics),
    )

    assert 200.0 < diagnostics.temperature.asnumpy().mean() < 320.0
    assert (diagnostics.pressure.asnumpy() > 0).all()
    # pressure grows downward: the surface interface above the top full level
    assert (diagnostics.pressure_ifc.asnumpy()[:, -1] > diagnostics.pressure.asnumpy()[:, 0]).all()
    for name, field_ in fw.fields(prognostic).items():
        np.testing.assert_array_equal(field_.asnumpy(), before[name], err_msg=name)
    assert tracer_state.qv is not None
    np.testing.assert_array_equal(tracer_state.qv.asnumpy(), 1e-3)


def test_run_rejects_a_tendency_of_no_advanced_field(grid: base_grid.Grid) -> None:
    process, _ = recording(VnTendencyOutput, {"tend_vn": 1e-4}, time_control(), grid)
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    with pytest.raises(ValueError, match="'tend_vn' is not the tendency of an advanced field"):
        physics.run(
            make_input(prognostic, tracer_state), output=make_output(prognostic, tracer_state)
        )


def test_run_updates_the_perturbed_exner_when_given(grid: base_grid.Grid) -> None:
    process, _ = recording(ThermoOutput, {"tend_temperature": 1e-3}, time_control(), grid)
    physics = driver(grid, [process])
    prognostic, tracer_state = prognostics(grid), tracers(grid)
    exner_before = prognostic.exner.asnumpy().copy()
    perturbed_exner = fw.zeros(qty.PerturbedExnerOnCellK, grid, None)

    physics.run(
        make_input(prognostic, tracer_state),
        output=make_output(prognostic, tracer_state, perturbed_exner=perturbed_exner),
    )

    # the perturbed exner follows the exner increment
    np.testing.assert_allclose(
        perturbed_exner.asnumpy(), prognostic.exner.asnumpy() - exner_before, atol=1e-15
    )
    assert np.any(perturbed_exner.asnumpy() != 0.0)


def test_run_syncs_the_halos_at_entry_and_of_what_it_wrote(grid: base_grid.Grid) -> None:
    exchange = RecordingExchange()
    process, recorder = recording(ThermoOutput, {"tend_temperature": 1e-3}, time_control(), grid)
    physics = driver(grid, [process], exchange=exchange)
    prognostic, tracer_state = prognostics(grid), tracers(grid)
    perturbed_exner = fw.zeros(qty.PerturbedExnerOnCellK, grid, None)

    physics.run(
        make_input(prognostic, tracer_state),
        output=make_output(prognostic, tracer_state, perturbed_exner=perturbed_exner),
    )

    # identity, not equality: the syncs must target the very buffers that were read and
    # written; no wind tendency was produced, so vn was neither written nor synced
    (state,) = recorder.states
    assert exchange.events == [
        (dims.CellDim, [id(prognostic.rho), id(state.u), id(state.v), id(prognostic.w)]),
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
    ]


def test_run_syncs_the_wind_sums_before_projecting_them_onto_vn(grid: base_grid.Grid) -> None:
    exchange = RecordingExchange()
    process, _ = recording(WindOutput, {"tend_u": 1e-4}, time_control(), grid)
    physics = driver(grid, [process], exchange=exchange)
    prognostic, tracer_state = prognostics(grid), tracers(grid)

    physics.run(make_input(prognostic, tracer_state), output=make_output(prognostic, tracer_state))

    sums = physics.accumulators
    assert exchange.events[1] == (dims.CellDim, [id(sums["tend_u"]), id(sums["tend_v"])])
    assert exchange.events[-1] == (dims.EdgeDim, [id(prognostic.vn)])


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
    with pytest.raises(ValueError, match="'sensible_heat_flux' is provided by more than one"):
        driver(grid, [first, second])


def test_diagnostic_fields_are_keyed_by_standard_name_and_skip_the_tendencies(
    grid: base_grid.Grid,
) -> None:
    process, _ = recording(FluxOutput, {}, time_control(), grid)
    physics = driver(grid, [process])

    ((name, field_),) = physics.diagnostic_fields().items()

    assert name == qty.SensibleHeatFluxOnCell.standard_name
    assert field_ is physics._process_outputs[process].sensible_heat_flux


class PrecipitationOutput(fw.State):
    pflx: field.Field[qty.PrecipitationFluxOnCellK]


def test_a_diagnostic_offered_by_two_processes_is_rejected(grid: base_grid.Grid) -> None:
    first, _ = recording(PrecipitationOutput, {}, time_control(), grid)
    second, _ = recording(PrecipitationOutput, {}, time_control(), grid)
    physics = driver(grid, [first, second])

    with pytest.raises(ValueError, match="'precipitation_flux' is an output of more than one"):
        physics.diagnostic_fields()
