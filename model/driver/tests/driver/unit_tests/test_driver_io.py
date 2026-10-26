# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the driver-side IO bridge (``driver_io``).

These tests are data-free: they use the ``simple_grid`` and a zero-initialised
``PrognosticState`` so they need no serialized/grid test data.
"""

import copy
import dataclasses
import datetime
import pathlib
import uuid

import gt4py.next as gtx
import numpy as np
import pytest
import xarray as xr

import icon4py.model.common.exceptions as errors
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.common import dimension as dims, time, type_alias as ta
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import base, simple, vertical as v_grid
from icon4py.model.common.interpolation import interpolation_attributes as intp_attr
from icon4py.model.common.io import io as common_io
from icon4py.model.common.metrics import metrics_attributes as metrics_attr
from icon4py.model.common.states import (
    data as state_data,
    diagnostic_state as diagnostics,
    model as state_model,
    prognostic_state as prognostics,
    tracer_states,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import driver_io
from icon4py.model.testing import test_utils
from icon4py.model.testing.fixtures import backend

from ..fixtures import ALL_OUTPUT_VARIABLES


@pytest.fixture
def grid() -> base.Grid:
    return simple.simple_grid()


def _make_prognostic_state(
    grid: base.Grid, allocator: gtx.typing.Backend | None = None
) -> prognostics.PrognosticState:
    # Constructed directly (instead of `initialize_prognostic_state`) so it works with
    # the generic `simple_grid`.
    def _cell_k() -> gtx.Field:
        return data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )

    def _cell_khalf() -> gtx.Field:
        return data_alloc.zero_field(
            grid, dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat, allocator=allocator
        )

    return prognostics.PrognosticState(
        rho=_cell_k(),
        w=_cell_khalf(),
        vn=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        ),
        exner=_cell_k(),
        theta_v=_cell_k(),
    )


@pytest.fixture
def prognostic_state(grid: base.Grid) -> prognostics.PrognosticState:
    return _make_prognostic_state(grid)


def _expected(
    cf_key: str, horizontal_dim: gtx.Dimension, *, vertical_dim: gtx.Dimension = dims.KDim
) -> state_model.FieldMetaData:
    """Expected output metadata: the shared CF entry plus the expected dims.
    ``standard_name``/``units`` come from the shared table rather than being
    re-spelled here; ``dims`` is stated independently of the production code so
    the assertions stay a genuine check. A field on interface levels is expected
    on ``KHalfDim``, which is what decides the netCDF vertical dimension name."""
    return dataclasses.replace(
        state_data.PROGNOSTIC_CF_ATTRIBUTES[cf_key],
        dims=(horizontal_dim, vertical_dim),
    )


#: Expected output metadata per output variable (keyed by CF name).
_EXPECTED: dict[str, state_model.FieldMetaData] = {
    "air_density": _expected("air_density", dims.CellDim),
    "exner_function": _expected("exner_function", dims.CellDim),
    "virtual_potential_temperature": _expected("virtual_potential_temperature", dims.CellDim),
    "upward_air_velocity": _expected(
        "upward_air_velocity", dims.CellDim, vertical_dim=dims.KHalfDim
    ),
    "normal_velocity": _expected("normal_velocity", dims.EdgeDim),
}

#: UGRID dimension names of the horizontal dimensions.
_UGRID_DIM_NAMES: dict[gtx.Dimension, str] = {
    dims.CellDim: "cell",
    dims.EdgeDim: "edge",
    dims.VertexDim: "vertex",
}


def _horizontal_size(grid: base.Grid, dim: gtx.Dimension) -> int:
    return {
        dims.CellDim: grid.num_cells,
        dims.EdgeDim: grid.num_edges,
        dims.VertexDim: grid.num_vertices,
    }[dim]


def test_assembles_all_default_variables(
    prognostic_state: prognostics.PrognosticState, grid: base.Grid
) -> None:
    state = driver_io.prognostic_state_to_dataarrays(prognostic_state)

    assert set(state.keys()) == set(driver_io.PROGNOSTIC_VARIABLES)
    for name, da in state.items():
        assert isinstance(da, xr.DataArray)
        expected = _EXPECTED[name]
        assert expected.dims is not None
        horizontal_dim = next(d for d in expected.dims if d.kind == gtx.DimensionKind.HORIZONTAL)
        on_half_levels = dims.KHalfDim in expected.dims

        vertical_name = "half_level" if on_half_levels else "level"
        assert da.dims == (_UGRID_DIM_NAMES[horizontal_dim], vertical_name)

        vertical_size = grid.num_levels + 1 if on_half_levels else grid.num_levels
        assert da.shape == (_horizontal_size(grid, horizontal_dim), vertical_size)


def test_dataarrays_carry_cf_and_ugrid_metadata(
    prognostic_state: prognostics.PrognosticState,
) -> None:
    state = driver_io.prognostic_state_to_dataarrays(prognostic_state)

    air_density = state["air_density"]
    # CF metadata from states.data
    assert air_density.attrs["standard_name"] == "air_density"
    assert air_density.attrs["units"] == "kg m-3"
    # UGRID metadata added by io.utils.to_data_array for the horizontal dimension
    assert air_density.attrs["location"] == "face"
    assert air_density.attrs["mesh"] == "mesh"
    assert air_density.attrs["coordinates"] == "clon clat"

    # edge field gets the edge location mapping
    assert state["normal_velocity"].attrs["location"] == "edge"


def test_does_not_mutate_shared_cf_attributes(
    prognostic_state: prognostics.PrognosticState,
) -> None:
    """`to_data_array` adds UGRID keys to the attrs it is handed; the shared
    module-level CF attribute table must be left untouched."""
    before = copy.deepcopy(state_data.PROGNOSTIC_CF_ATTRIBUTES)

    driver_io.prognostic_state_to_dataarrays(prognostic_state)

    assert before == state_data.PROGNOSTIC_CF_ATTRIBUTES
    # specifically, no UGRID keys reach the attrs the shared table renders
    for entry in state_data.PROGNOSTIC_CF_ATTRIBUTES.values():
        rendered = entry.as_dict()
        assert "location" not in rendered
        assert "mesh" not in rendered
        assert "coordinates" not in rendered


def test_variables_subset(prognostic_state: prognostics.PrognosticState) -> None:
    subset = ["air_density", "normal_velocity"]
    state = driver_io.prognostic_state_to_dataarrays(prognostic_state, variables=subset)
    assert set(state.keys()) == set(subset)


def test_unknown_variable_raises(prognostic_state: prognostics.PrognosticState) -> None:
    with pytest.raises(ValueError, match="Unknown prognostic output variable"):
        driver_io.prognostic_state_to_dataarrays(prognostic_state, variables=["not_a_field"])


def test_data_is_host_numpy(
    grid: base.Grid,
    backend: gtx.typing.Backend | None,
) -> None:
    """The buffer handed to the writers must be a host numpy array, not a device array.

    Parameterized on the backend (``--backend``) so that with a GPU backend the inputs
    really are device buffers and the host transfer is exercised.
    """
    prognostic_state = _make_prognostic_state(grid, allocator=backend)
    state = driver_io.prognostic_state_to_dataarrays(prognostic_state)
    for da in state.values():
        assert isinstance(da.data, np.ndarray)


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
        for name in driver_io.DIAGNOSTIC_VARIABLES
    }
    state = driver_io.diagnostic_fields_to_dataarrays(diagnostic_fields)
    for da in state.values():
        assert isinstance(da.data, np.ndarray)


def test_create_io_monitor_passes_the_io_config(
    grid: base.Grid, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``IOMonitor`` is replaced by a recorder so the test needs no real grid file."""
    recorded: dict[str, object] = {}

    class _RecordingMonitor:
        def __init__(
            self,
            *,
            config: common_io.IOConfig,
            vertical_size: v_grid.VerticalGrid,
            horizontal_size: base.HorizontalGridSize,
            grid_file_name: pathlib.Path,
            grid_id: uuid.UUID,
            dtime: datetime.timedelta,
            process_props: decomposition_defs.ProcessProperties,
            decomposition_info: decomposition_defs.DecompositionInfo | None,
        ) -> None:
            recorded["config"] = config
            recorded["grid_file_name"] = grid_file_name
            recorded["grid_id"] = grid_id

    monkeypatch.setattr(common_io, "IOMonitor", _RecordingMonitor)
    io_config = common_io.IOConfig(
        output_path=tmp_path,
        field_groups=[common_io.FieldGroupIOConfig(basename="a", variables=["air_density"])],
    )

    driver_io.create_io_monitor(
        io_config=io_config,
        grid_file_path=tmp_path / "grid.nc",
        grid=grid,
        vertical_grid=None,  # type: ignore[arg-type] # not used by the recorder
        dtime=datetime.timedelta(seconds=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
        decomposition_info=None,
    )

    # the configuration carries the output directory
    assert recorded["config"] is io_config
    assert recorded["grid_file_name"] == tmp_path / "grid.nc"
    # the string grid id is converted to a UUID at the IO boundary
    assert recorded["grid_id"] == uuid.UUID(grid.id)


# --------------------------------------------------------------------------------------
# Output state assembly
# --------------------------------------------------------------------------------------


class _StubFieldSource:
    """Minimal FieldSource stand-in: serves pre-built fields by name."""

    def __init__(self, fields: dict[str, gtx.Field]) -> None:
        self._fields = fields

    def get(self, name: str) -> gtx.Field:
        return self._fields[name]


class _StubStaticFields:
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


def _diagnostics_computer(grid: base.Grid, variables: list[str]) -> driver_io.DiagnosticsComputer:
    static = _StubStaticFields(grid)
    return driver_io.DiagnosticsComputer(
        grid=grid,
        backend=None,
        variables=variables,
        metrics=static.metrics,  # type: ignore[arg-type]  # stand-in serving the static fields
        interpolation=static.interpolation,  # type: ignore[arg-type]  # stand-in serving the static fields
    )


def _metadata(key: str, *field_dims: gtx.Dimension) -> state_model.FieldMetaData:
    """Precipitation metadata with the dims the physics store allocates its buffer from."""
    return dataclasses.replace(state_data.PRECIPITATION_CF_ATTRIBUTES[key], dims=field_dims)


def _physics_diagnostics(
    grid: base.Grid, outputs: dict[str, dict[str, state_model.FieldMetaData]]
) -> physics_state.DiagnosticsStore:
    """A physics diagnostics store with the given outputs allocated, per process."""
    store = physics_state.DiagnosticsStore(grid=grid)
    for process_name, outputs_properties in outputs.items():
        store.allocate(process_name, outputs_properties)
    return store


def _rainfall_outputs() -> dict[str, dict[str, state_model.FieldMetaData]]:
    return {"muphys": {"pr": _metadata("rainfall_flux", dims.CellDim, dims.KDim)}}


def _io_config(*groups: tuple[str, list[str]]) -> common_io.IOConfig:
    return common_io.IOConfig(
        field_groups=[
            common_io.FieldGroupIOConfig(basename=basename, variables=variables)
            for basename, variables in groups
        ]
    )


def _assembler(
    grid: base.Grid,
    io_config: common_io.IOConfig,
    *,
    tracer_config: tracer_states.TracerConfig | None = None,
    physics_diagnostics: physics_state.DiagnosticsStore | None = None,
) -> driver_io.OutputStateAssembler:
    return driver_io.OutputStateAssembler(
        io_config=io_config,
        grid=grid,
        backend=None,
        static_field_factories=_StubStaticFields(grid),  # type: ignore[arg-type]  # stand-in serving the static fields
        tracer_config=tracer_config,
        physics_diagnostics=physics_diagnostics,
    )


def _diagnostic_state(grid: base.Grid) -> diagnostics.DiagnosticState:
    """The driver's diagnostic state the output diagnostics are computed into."""
    return diagnostics.initialize_diagnostic_state(grid, allocator=None)


def _uniform_prognostic_state(grid: base.Grid) -> prognostics.PrognosticState:
    """A thermodynamically plausible uniform state (the zero state gives no pressure)."""
    return prognostics.PrognosticState(
        rho=data_alloc.constant_field(grid, 1.2, dims.CellDim, dims.KDim),
        w=data_alloc.zero_field(grid, dims.CellDim, dims.KHalfDim, dtype=ta.wpfloat),
        vn=data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat),
        exner=data_alloc.constant_field(grid, 0.95, dims.CellDim, dims.KDim),
        theta_v=data_alloc.constant_field(grid, 300.0, dims.CellDim, dims.KDim),
    )


def test_available_variables_are_the_prognostics_and_diagnostics(grid: base.Grid) -> None:
    assembler = _assembler(grid, _io_config(("prognostics", ["air_density"])))
    assert assembler.available_variables == set(ALL_OUTPUT_VARIABLES)


def test_available_variables_include_the_active_tracers(grid: base.Grid) -> None:
    assembler = _assembler(
        grid,
        _io_config(("moisture", ["qv"])),
        tracer_config=tracer_states.TracerConfig(qv=True, qc=True),
    )
    assert assembler.available_variables == {*ALL_OUTPUT_VARIABLES, "qv", "qc"}


def test_available_variables_include_the_physics_diagnostics(grid: base.Grid) -> None:
    assembler = _assembler(
        grid,
        _io_config(("precipitation", ["rainfall_flux"])),
        physics_diagnostics=_physics_diagnostics(grid, _rainfall_outputs()),
    )
    assert assembler.available_variables == {*ALL_OUTPUT_VARIABLES, "rainfall_flux"}


def test_physics_diagnostic_outputs_are_named_by_standard_name(grid: base.Grid) -> None:
    outputs = driver_io.physics_diagnostic_outputs(_physics_diagnostics(grid, _rainfall_outputs()))
    assert set(outputs) == {"rainfall_flux"}
    assert outputs["rainfall_flux"].name == "pr"
    assert outputs["rainfall_flux"].process_name == "muphys"


def test_physics_diagnostic_outputs_include_surface_fields(grid: base.Grid) -> None:
    # a purely horizontal (surface) output is offered like the others
    outputs = driver_io.physics_diagnostic_outputs(
        _physics_diagnostics(
            grid,
            {
                "muphys": {
                    "pr": _metadata("rainfall_flux", dims.CellDim, dims.KDim),
                    "pr_sfc": _metadata("precipitation_flux", dims.CellDim),
                }
            },
        )
    )
    assert set(outputs) == {"rainfall_flux", "precipitation_flux"}
    assert outputs["precipitation_flux"].field.domain.dims == (dims.CellDim,)


def test_physics_diagnostic_outputs_without_physics_are_empty() -> None:
    assert driver_io.physics_diagnostic_outputs(None) == {}


def test_duplicate_physics_diagnostic_names_raise(grid: base.Grid) -> None:
    rainfall = _metadata("rainfall_flux", dims.CellDim, dims.KDim)
    store = _physics_diagnostics(grid, {"muphys": {"pr": rainfall}, "graupel": {"rain": rainfall}})
    with pytest.raises(ValueError, match="more than one process: muphys, graupel"):
        driver_io.physics_diagnostic_outputs(store)


def test_validate_output_variables_rejects_unknown_names() -> None:
    io_config = _io_config(("winds", ["eastward_wind", "foo", "bar"]))
    sources = driver_io.output_variable_sources(tracer_config=None, physics_outputs={})
    with pytest.raises(errors.InvalidConfigError, match="field group 'winds': bar, foo") as err:
        driver_io.validate_output_variables(io_config, sources)
    # the message tells the user what is available
    assert "air_density" in str(err.value)


def test_validate_output_variables_rejects_ambiguous_names_only_when_requested(
    grid: base.Grid,
) -> None:
    # a physics output whose standard name is a prognostic variable name
    clashing = state_model.FieldMetaData(
        standard_name="air_density", units="kg m-3", dims=(dims.CellDim, dims.KDim)
    )
    sources = driver_io.output_variable_sources(
        tracer_config=None,
        physics_outputs=driver_io.physics_diagnostic_outputs(
            _physics_diagnostics(grid, {"muphys": {"rho": clashing}})
        ),
    )
    assert sources["air_density"] == ["prognostic", "physics"]

    driver_io.validate_output_variables(_io_config(("exner", ["exner_function"])), sources)
    with pytest.raises(errors.InvalidConfigError, match="ambiguous: provided by the prognostic"):
        driver_io.validate_output_variables(_io_config(("rho", ["air_density"])), sources)


def test_assembler_rejects_unknown_variables_at_setup(grid: base.Grid) -> None:
    # no tracer is active, so 'qv' is not an output variable of this run
    with pytest.raises(errors.InvalidConfigError, match="field group 'moisture': qv"):
        _assembler(grid, _io_config(("moisture", ["qv"])))


def test_assembler_assembles_only_the_requested_variables(grid: base.Grid) -> None:
    prognostic_state = _uniform_prognostic_state(grid)
    tracer_state = tracer_states.TracerState(
        qv=data_alloc.constant_field(grid, 0.01, dims.CellDim, dims.KDim)
    )
    assembler = _assembler(
        grid,
        _io_config(
            ("prognostics", ["air_density", "eastward_wind", "surface_pressure"]),
            ("moisture", ["qv", "rainfall_flux"]),
        ),
        tracer_config=tracer_states.TracerConfig(qv=True),
        physics_diagnostics=_physics_diagnostics(grid, _rainfall_outputs()),
    )

    diagnostic_state = _diagnostic_state(grid)
    state = assembler.assemble(
        {"air_density", "qv", "rainfall_flux", "surface_pressure"},
        prognostic_state,
        tracer_state,
        diagnostic_state,
    )
    assert set(state) == {"air_density", "qv", "rainfall_flux", "surface_pressure"}
    assert state["qv"].attrs["standard_name"] == "specific_humidity"
    assert state["rainfall_flux"].attrs["standard_name"] == "rainfall_flux"
    assert state["rainfall_flux"].dims == ("cell", "level")
    # a purely horizontal diagnostic
    assert state["surface_pressure"].dims == ("cell",)

    winds = assembler.assemble({"eastward_wind"}, prognostic_state, tracer_state, diagnostic_state)
    assert set(winds) == {"eastward_wind"}
    # the diagnostics are computed into the driver's diagnostic state
    assert winds["eastward_wind"].data is diagnostic_state.u.ndarray


def test_assembler_builds_no_diagnostics_computer_without_diagnostics(
    grid: base.Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Unexpected:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("no diagnostic is requested: the computer must not be built")

    monkeypatch.setattr(driver_io, "DiagnosticsComputer", _Unexpected)
    _assembler(grid, _io_config(("prognostics", ["air_density", "normal_velocity"])))


def test_diagnostics_computer_returns_only_the_requested_variables(grid: base.Grid) -> None:
    computer = _diagnostics_computer(grid, ["pressure", "eastward_wind"])
    prognostic_state = _uniform_prognostic_state(grid)
    diagnostic_state = _diagnostic_state(grid)

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
    diagnostic_state = _diagnostic_state(grid)

    fields = computer.compute(
        _uniform_prognostic_state(grid), tracer_states.TracerState(), diagnostic_state, requested
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
        _uniform_prognostic_state(grid),
        tracer_states.TracerState(),
        _diagnostic_state(grid),
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
        _uniform_prognostic_state(grid),
        tracer_states.TracerState(),
        _diagnostic_state(grid),
        ["eastward_wind"],
    )

    assert set(fields) == {"eastward_wind"}


def test_diagnostics_computer_uses_the_active_tracers(grid: base.Grid) -> None:
    """Moist air: the temperature falls below the virtual temperature; dry air: they agree."""
    requested = ["temperature", "virtual_temperature"]
    computer = _diagnostics_computer(grid, requested)
    prognostic_state = _uniform_prognostic_state(grid)
    diagnostic_state = _diagnostic_state(grid)

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


def test_tracer_state_to_dataarrays(grid: base.Grid, backend: gtx.typing.Backend | None) -> None:
    tracer_state = tracer_states.TracerState(
        qv=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend)
    )

    state = driver_io.tracer_state_to_dataarrays(tracer_state, ["qv"])

    assert set(state) == {"qv"}
    assert state["qv"].dims == ("cell", "level")
    assert state["qv"].attrs["standard_name"] == "specific_humidity"
    assert state["qv"].attrs["location"] == "face"
    # host numpy, also from a device buffer
    assert isinstance(state["qv"].data, np.ndarray)
    with pytest.raises(ValueError, match="'qc' is not active"):
        driver_io.tracer_state_to_dataarrays(tracer_state, ["qc"])
    with pytest.raises(ValueError, match="Unknown tracer output variable 'foo'"):
        driver_io.tracer_state_to_dataarrays(tracer_state, ["foo"])


def test_diagnostic_fields_to_dataarrays(grid: base.Grid) -> None:
    """The diagnostic assembly mirrors the prognostic one: correct dims/metadata, host
    numpy buffers, and the shared CF table is not mutated."""
    before = copy.deepcopy(state_data.DIAGNOSTIC_CF_ATTRIBUTES)

    # cell/full-level fields, like what compute_diagnostics returns
    def _zero_cell_k() -> gtx.Field:
        return data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

    fields = {name: _zero_cell_k() for name in driver_io.DIAGNOSTIC_VARIABLES}
    # the surface pressure is a cell field
    fields["surface_pressure"] = data_alloc.zero_field(grid, dims.CellDim, dtype=ta.wpfloat)
    state = driver_io.diagnostic_fields_to_dataarrays(fields)

    assert set(state.keys()) == set(driver_io.DIAGNOSTIC_VARIABLES)
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
