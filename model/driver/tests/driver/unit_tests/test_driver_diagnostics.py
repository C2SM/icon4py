# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the output diagnostics (``driver_diagnostics``), data-free on the simple grid."""

import copy

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, simple
from icon4py.model.common.interpolation import interpolation_attributes as intp_attr
from icon4py.model.common.metrics import metrics_attributes as metrics_attr
from icon4py.model.common.states import (
    data as state_data,
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_states,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import driver_diagnostics
from icon4py.model.testing import test_utils
from icon4py.model.testing.fixtures import backend


@pytest.fixture
def grid() -> base.Grid:
    return simple.simple_grid()


class _StubFieldSource:
    """Minimal FieldSource stand-in: serves pre-built fields by name."""

    def __init__(self, fields: dict[str, gtx.Field]) -> None:
        self._fields = fields

    def get(self, name: str) -> gtx.Field:
        return self._fields[name]


class StubStaticFields:
    """The two field sources the diagnostics computer reads its static inputs from."""

    def __init__(self, grid: base.Grid) -> None:
        self.metrics = _StubFieldSource(
            {
                metrics_attr.DDQZ_Z_FULL: data_alloc.constant_field(
                    grid, 100.0, dims.CellDim, dims.KDim
                ),
            }
        )
        self.interpolation = _StubFieldSource(
            {
                intp_attr.RBF_VEC_COEFF_C1: data_alloc.zero_field(
                    grid, dims.CellDim, dims.C2E2C2EDim
                ),
                intp_attr.RBF_VEC_COEFF_C2: data_alloc.zero_field(
                    grid, dims.CellDim, dims.C2E2C2EDim
                ),
            }
        )


def _diagnostics_computer(
    grid: base.Grid, variables: list[str]
) -> driver_diagnostics.DiagnosticsComputer:
    static = StubStaticFields(grid)
    return driver_diagnostics.DiagnosticsComputer(
        grid=grid,
        backend=None,
        variables=variables,
        metrics=static.metrics,  # type: ignore[arg-type]  # stand-in serving the static fields
        interpolation=static.interpolation,  # type: ignore[arg-type]  # stand-in serving the static fields
    )


def make_diagnostic_state(grid: base.Grid) -> diagnostics.DiagnosticState:
    """The driver's diagnostic state the output diagnostics are computed into."""
    return diagnostics.initialize_diagnostic_state(grid, allocator=None)


def uniform_prognostic_state(grid: base.Grid) -> prognostics.PrognosticState:
    """A thermodynamically plausible uniform state (the zero state gives no pressure)."""
    return prognostics.PrognosticState(
        rho=data_alloc.constant_field(grid, 1.2, dims.CellDim, dims.KDim),
        w=data_alloc.zero_field(grid, dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat),
        vn=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat),
        exner=data_alloc.constant_field(grid, 0.95, dims.CellDim, dims.KDim),
        theta_v=data_alloc.constant_field(grid, 300.0, dims.CellDim, dims.KDim),
    )


def test_diagnostic_data_is_host_numpy(
    grid: base.Grid,
    backend: gtx.typing.Backend | None,
) -> None:
    """Same host-transfer guarantee for the diagnostic assembly path.

    The diagnostic fields come from the (device-resident, on GPU backends) buffers of
    the ``DiagnosticsComputer``; assembling them must also land host numpy arrays.
    """
    diagnostic_fields = {
        name: data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
        )
        for name in driver_diagnostics.DIAGNOSTIC_VARIABLES
    }
    state = driver_diagnostics.diagnostic_fields_to_dataarrays(diagnostic_fields)
    for da in state.values():
        assert isinstance(da.data, np.ndarray)


def test_diagnostics_computer_returns_only_the_requested_variables(grid: base.Grid) -> None:
    computer = _diagnostics_computer(grid, ["pressure", "eastward_wind"])
    prognostic_state = uniform_prognostic_state(grid)
    diagnostic_state = make_diagnostic_state(grid)

    fields = computer.compute(
        prognostic_state, tracer_states.TracerState(), diagnostic_state, ["pressure"]
    )
    assert set(fields) == {"pressure"}
    assert fields["pressure"] is diagnostic_state.pressure
    assert (fields["pressure"].asnumpy() > 0.0).all()
    # the wind program did not run: the diagnostic state's winds are untouched
    assert not diagnostic_state.u.asnumpy().any()

    with pytest.raises(ValueError, match="not configured for this computer: temperature"):
        computer.compute(
            prognostic_state, tracer_states.TracerState(), diagnostic_state, ["temperature"]
        )


def test_diagnostics_computer_rejects_unknown_variables(grid: base.Grid) -> None:
    with pytest.raises(ValueError, match="Unknown diagnostic output variables: foo"):
        _diagnostics_computer(grid, ["foo"])


def test_diagnostics_computer_surface_pressure(grid: base.Grid) -> None:
    """The surface pressure is a cell field: the bottom half-level row of the pressure program."""
    requested = ["pressure", "surface_pressure"]
    computer = _diagnostics_computer(grid, requested)
    diagnostic_state = make_diagnostic_state(grid)

    fields = computer.compute(
        uniform_prognostic_state(grid), tracer_states.TracerState(), diagnostic_state, requested
    )

    surface_pressure = fields["surface_pressure"]
    assert surface_pressure.domain.dims == (dims.CellDim,)
    assert surface_pressure.asnumpy().shape == (grid.num_cells,)
    # the diagnostic state's view of the bottom row of its half-level pressure: no copy
    assert np.shares_memory(surface_pressure.ndarray, diagnostic_state.pressure_ifc.ndarray)
    test_utils.assert_dallclose(
        surface_pressure.asnumpy(), diagnostic_state.pressure_ifc.asnumpy()[:, -1]
    )
    # hydrostatic: the surface pressure exceeds the pressure of the lowest model level
    assert (surface_pressure.asnumpy() > fields["pressure"].asnumpy()[:, -1]).all()
    # the surface pressure alone also runs the temperature and pressure programs
    alone = _diagnostics_computer(grid, ["surface_pressure"]).compute(
        uniform_prognostic_state(grid),
        tracer_states.TracerState(),
        make_diagnostic_state(grid),
        ["surface_pressure"],
    )
    assert set(alone) == {"surface_pressure"}
    test_utils.assert_dallclose(alone["surface_pressure"].asnumpy(), surface_pressure.asnumpy())


def test_diagnostics_computer_reads_only_the_requested_fields(
    grid: base.Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call for the winds does not touch the other diagnostics of the state."""

    def _not_requested(self: diagnostics.DiagnosticState) -> gtx.Field:
        raise AssertionError("the surface pressure was not requested")

    monkeypatch.setattr(diagnostics.DiagnosticState, "surface_pressure", property(_not_requested))
    computer = _diagnostics_computer(grid, ["eastward_wind", "surface_pressure"])

    fields = computer.compute(
        uniform_prognostic_state(grid),
        tracer_states.TracerState(),
        make_diagnostic_state(grid),
        ["eastward_wind"],
    )

    assert set(fields) == {"eastward_wind"}


def test_diagnostics_computer_uses_the_active_tracers(grid: base.Grid) -> None:
    """Moist air: the temperature falls below the virtual temperature; dry air: they agree."""
    requested = ["temperature", "virtual_temperature"]
    computer = _diagnostics_computer(grid, requested)
    prognostic_state = uniform_prognostic_state(grid)
    diagnostic_state = make_diagnostic_state(grid)

    dry = computer.compute(
        prognostic_state, tracer_states.TracerState(), diagnostic_state, requested
    )
    test_utils.assert_dallclose(dry["temperature"].asnumpy(), dry["virtual_temperature"].asnumpy())
    dry_temperature = dry["temperature"].asnumpy().copy()

    moist_tracers = tracer_states.TracerState(
        qv=data_alloc.constant_field(grid, 0.01, dims.CellDim, dims.KDim)
    )
    moist = computer.compute(prognostic_state, moist_tracers, diagnostic_state, requested)
    test_utils.assert_dallclose(moist["virtual_temperature"].asnumpy(), dry_temperature)
    assert (moist["temperature"].asnumpy() < moist["virtual_temperature"].asnumpy()).all()


def test_diagnostic_fields_to_dataarrays(grid: base.Grid) -> None:
    """The diagnostic assembly mirrors the prognostic one: correct dims/metadata, host
    numpy buffers, and the shared CF table is not mutated."""
    before = copy.deepcopy(state_data.DIAGNOSTIC_CF_ATTRIBUTES)

    # cell/full-level fields, like what compute_diagnostics returns
    def _zero_cell_k() -> gtx.Field:
        return data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

    fields = {name: _zero_cell_k() for name in driver_diagnostics.DIAGNOSTIC_VARIABLES}
    # the surface pressure is a cell field
    fields["surface_pressure"] = data_alloc.zero_field(grid, dims.CellDim, dtype=ta.wpfloat)
    state = driver_diagnostics.diagnostic_fields_to_dataarrays(fields)

    assert set(state.keys()) == set(driver_diagnostics.DIAGNOSTIC_VARIABLES)
    for name, da in state.items():
        if name == "surface_pressure":
            assert da.dims == ("cell",)
            assert da.shape == (grid.num_cells,)
        else:
            assert da.dims == ("cell", "level")
            assert da.shape == (grid.num_cells, grid.num_levels)
        assert isinstance(da.data, np.ndarray)

    assert state["surface_pressure"].attrs["standard_name"] == "air_pressure_at_ground_level"
    assert state["temperature"].attrs["standard_name"] == "air_temperature"
    assert state["pressure"].attrs["units"] == "Pa"
    assert state["eastward_wind"].attrs["location"] == "face"

    # shared diagnostic CF table must be untouched
    assert before == state_data.DIAGNOSTIC_CF_ATTRIBUTES
