# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


"""Unit tests for the output diagnostics (``driver_diagnostics``), data-free on the simple grid."""

from collections.abc import Mapping

import gt4py.next as gtx
import numpy as np
import pytest
import xarray as xr

from icon4py.model.common import dimension as dims, field, framework as fw
from icon4py.model.common.framework import quantities as qty, states
from icon4py.model.common.grid import base, simple
from icon4py.model.common.interpolation import interpolation_attributes as intp_attr
from icon4py.model.common.io import utils as io_utils
from icon4py.model.common.metrics import metrics_attributes as metrics_attr
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import driver_diagnostics, driver_output_sources as output_sources
from icon4py.model.driver.driver_diagnostics import (
    EASTWARD_WIND,
    NORTHWARD_WIND,
    PRESSURE,
    SURFACE_PRESSURE,
    TEMPERATURE,
    VIRTUAL_TEMPERATURE,
)
from icon4py.model.testing import test_utils
from icon4py.model.testing.fixtures import backend


def _dataarrays(fields: Mapping[str, field.Field]) -> dict[str, xr.DataArray]:
    """The fields as IO writes them."""
    return {name: io_utils.to_data_array(field_, to_host=True) for name, field_ in fields.items()}


@pytest.fixture
def grid() -> base.Grid:
    return simple.simple_grid()


#: Uniform wind-program inputs: with constant ``vn`` and RBF coefficients the reconstructed
#: cell winds are the coefficient times ``vn``, summed over the C2E2C2E neighbors.
VN = 10.0
RBF_COEFF_C1 = 0.1
RBF_COEFF_C2 = 0.2


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
                intp_attr.RBF_VEC_COEFF_C1: data_alloc.constant_field(
                    grid, RBF_COEFF_C1, dims.CellDim, dims.C2E2C2EDim
                ),
                intp_attr.RBF_VEC_COEFF_C2: data_alloc.constant_field(
                    grid, RBF_COEFF_C2, dims.CellDim, dims.C2E2C2EDim
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


def make_diagnostic_state(grid: base.Grid) -> states.Diagnostics:
    """The driver's diagnostic state the output diagnostics are computed into."""
    return fw.allocate(states.Diagnostics, grid, None)


def uniform_prognostic_state(grid: base.Grid) -> states.PrognosticState:
    """A thermodynamically plausible uniform state (the zero state gives no pressure)."""
    values = {"rho": 1.2, "w": 0.5, "vn": VN, "exner": 0.95, "theta_v": 300.0}
    return fw.allocate(states.PrognosticState, grid, None, fill=lambda name, _: values[name])


def test_diagnostic_data_is_host_numpy(
    grid: base.Grid,
    backend: gtx.typing.Backend | None,
) -> None:
    """
    The diagnostic fields come from the (device-resident, on GPU backends) buffers of the
    driver's diagnostics; assembling them must land host numpy arrays.
    """
    diagnostic_state = fw.allocate(states.Diagnostics, grid, backend)
    state = _dataarrays(output_sources.by_standard_name(diagnostic_state))
    assert state
    for da in state.values():
        assert isinstance(da.data, np.ndarray)


def test_diagnostics_computer_returns_only_the_requested_variables(grid: base.Grid) -> None:
    computer = _diagnostics_computer(grid, [PRESSURE, EASTWARD_WIND])
    prognostic_state = uniform_prognostic_state(grid)
    diagnostic_state = make_diagnostic_state(grid)

    fields = computer.compute(prognostic_state, states.TracerState(), diagnostic_state, [PRESSURE])
    assert set(fields) == {PRESSURE}
    assert fields[PRESSURE] is diagnostic_state.pressure
    assert (fields[PRESSURE].asnumpy() > 0.0).all()
    # the wind program did not run: the diagnostic state's winds are untouched
    assert not diagnostic_state.u.asnumpy().any()

    with pytest.raises(ValueError, match=f"not configured for this computer: {TEMPERATURE}"):
        computer.compute(prognostic_state, states.TracerState(), diagnostic_state, [TEMPERATURE])


def test_diagnostics_computer_winds(grid: base.Grid) -> None:
    """Constant ``vn`` and coefficients: each wind is their product summed over the neighbors."""
    requested = [EASTWARD_WIND, NORTHWARD_WIND]
    fields = _diagnostics_computer(grid, requested).compute(
        uniform_prognostic_state(grid),
        states.TracerState(),
        make_diagnostic_state(grid),
        requested,
    )
    neighbors = grid.get_connectivity(dims.C2E2C2E).asnumpy().shape[1]
    test_utils.assert_dallclose(fields[EASTWARD_WIND].asnumpy(), neighbors * RBF_COEFF_C1 * VN)
    test_utils.assert_dallclose(fields[NORTHWARD_WIND].asnumpy(), neighbors * RBF_COEFF_C2 * VN)


def test_diagnostics_computer_rejects_unknown_variables(grid: base.Grid) -> None:
    with pytest.raises(ValueError, match="Unknown diagnostic output variables: foo"):
        _diagnostics_computer(grid, ["foo"])


def test_diagnostics_computer_surface_pressure(grid: base.Grid) -> None:
    """The surface pressure is a cell field: the bottom half-level row of the pressure program."""
    requested = [PRESSURE, SURFACE_PRESSURE]
    computer = _diagnostics_computer(grid, requested)
    diagnostic_state = make_diagnostic_state(grid)

    fields = computer.compute(
        uniform_prognostic_state(grid), states.TracerState(), diagnostic_state, requested
    )

    surface_pressure = fields[SURFACE_PRESSURE]
    assert surface_pressure.quantity is qty.SurfacePressureOnCell
    assert surface_pressure.domain.dims == (dims.CellDim,)
    assert surface_pressure.asnumpy().shape == (grid.num_cells,)
    # a view of the bottom row of the half-level pressure: no copy
    assert np.shares_memory(surface_pressure.ndarray, diagnostic_state.pressure_ifc.ndarray)
    test_utils.assert_dallclose(
        surface_pressure.asnumpy(), diagnostic_state.pressure_ifc.asnumpy()[:, -1]
    )
    # hydrostatic: the surface pressure exceeds the pressure of the lowest model level
    assert (surface_pressure.asnumpy() > fields[PRESSURE].asnumpy()[:, -1]).all()
    # the surface pressure alone also runs the temperature and pressure programs
    alone = _diagnostics_computer(grid, [SURFACE_PRESSURE]).compute(
        uniform_prognostic_state(grid),
        states.TracerState(),
        make_diagnostic_state(grid),
        [SURFACE_PRESSURE],
    )
    assert set(alone) == {SURFACE_PRESSURE}
    test_utils.assert_dallclose(alone[SURFACE_PRESSURE].asnumpy(), surface_pressure.asnumpy())


def test_diagnostics_computer_runs_only_the_programs_of_the_requested_variables(
    grid: base.Grid,
) -> None:
    """A call for the winds does not run the pressure program."""
    computer = _diagnostics_computer(grid, [EASTWARD_WIND, SURFACE_PRESSURE])
    diagnostic_state = make_diagnostic_state(grid)

    fields = computer.compute(
        uniform_prognostic_state(grid), states.TracerState(), diagnostic_state, [EASTWARD_WIND]
    )

    assert set(fields) == {EASTWARD_WIND}
    assert not diagnostic_state.pressure_ifc.asnumpy().any()


def test_diagnostics_computer_uses_the_active_tracers(grid: base.Grid) -> None:
    """Moist air: the temperature falls below the virtual temperature; dry air: they agree."""
    requested = [TEMPERATURE, VIRTUAL_TEMPERATURE]
    computer = _diagnostics_computer(grid, requested)
    prognostic_state = uniform_prognostic_state(grid)
    diagnostic_state = make_diagnostic_state(grid)

    dry = computer.compute(prognostic_state, states.TracerState(), diagnostic_state, requested)
    test_utils.assert_dallclose(dry[TEMPERATURE].asnumpy(), dry[VIRTUAL_TEMPERATURE].asnumpy())
    dry_temperature = dry[TEMPERATURE].asnumpy().copy()

    moist_tracers = states.TracerState(
        qv=field.Field(
            qty.QvOnCellK, data_alloc.constant_field(grid, 0.01, dims.CellDim, dims.KDim)
        )
    )
    moist = computer.compute(prognostic_state, moist_tracers, diagnostic_state, requested)
    test_utils.assert_dallclose(moist[VIRTUAL_TEMPERATURE].asnumpy(), dry_temperature)
    assert (moist[TEMPERATURE].asnumpy() < moist[VIRTUAL_TEMPERATURE].asnumpy()).all()


def test_diagnostic_dataarrays_carry_dims_and_attributes(grid: base.Grid) -> None:
    """The diagnostics' DataArrays: correct dims and CF attributes, from their quantities."""
    variables = list(driver_diagnostics.DIAGNOSTIC_VARIABLES)
    fields = _diagnostics_computer(grid, variables).compute(
        uniform_prognostic_state(grid), states.TracerState(), make_diagnostic_state(grid), variables
    )

    state = _dataarrays(fields)

    assert set(state) == set(variables)
    for name, da in state.items():
        if name == SURFACE_PRESSURE:
            assert da.dims == ("cell",)
            assert da.shape == (grid.num_cells,)
        else:
            assert da.dims == ("cell", "level")
            assert da.shape == (grid.num_cells, grid.num_levels)
        assert isinstance(da.data, np.ndarray)
        assert da.attrs["standard_name"] == name
        assert "icon_var_name" not in da.attrs
    assert state[PRESSURE].attrs["units"] == "Pa"
    assert state[EASTWARD_WIND].attrs["location"] == "face"
