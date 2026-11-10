# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the driver-side IO bridge (``driver_io``), data-free on the simple grid."""

import dataclasses
import datetime
import pathlib
import uuid
from collections.abc import Iterable

import numpy as np
import pytest

import icon4py.model.common.exceptions as errors
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition_defs, mpi_decomposition
from icon4py.model.common.grid import base, simple, vertical as v_grid
from icon4py.model.common.io import io as common_io, io_servers
from icon4py.model.common.states import model as state_model, tracer_states
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import driver_diagnostics, driver_io, driver_output_sources

from ..fixtures import ALL_OUTPUT_VARIABLES
from .test_driver_diagnostics import (
    StubStaticFields,
    make_diagnostic_state,
    uniform_prognostic_state,
)
from .test_driver_output_sources import allocated_physics_outputs, rainfall_outputs


@pytest.fixture
def grid() -> base.Grid:
    return simple.simple_grid()


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
        io_ranks=None,
    )

    # the configuration carries the output directory
    assert recorded["config"] is io_config
    assert recorded["grid_file_name"] == tmp_path / "grid.nc"
    # the string grid id is converted to a UUID at the IO boundary
    assert recorded["grid_id"] == uuid.UUID(grid.id)


def test_create_io_monitor_builds_a_client_with_io_ranks(
    grid: base.Grid, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With dedicated IO ranks the compute side gets an ``IOClient`` on the intercomm.

    ``IOClient`` is replaced by a recorder (its construction sends the setup); the
    configuration handed to it is the one ``IOMonitor`` would get.
    """
    recorded: dict[str, object] = {}

    class _RecordingClient:
        def __init__(
            self,
            *,
            config: common_io.IOConfig,
            dtime: datetime.timedelta,
            grid_id: uuid.UUID,
            horizontal_size: base.HorizontalGridSize,
            decomposition_info: decomposition_defs.DecompositionInfo,
            intercomm: object,
            io_rank: int,
        ) -> None:
            recorded["config"] = config
            recorded["grid_id"] = grid_id
            recorded["horizontal_size"] = horizontal_size
            recorded["intercomm"] = intercomm
            recorded["decomposition_info"] = decomposition_info
            recorded["io_rank"] = io_rank

    monkeypatch.setattr(io_servers, "IOClient", _RecordingClient)
    intercomm = object()
    io_ranks = mpi_decomposition.IORankSplit(
        is_io_rank=False,
        process_props=None,  # type: ignore[arg-type]  # not used by the recorder
        intercomm=intercomm,  # type: ignore[arg-type]  # not used by the recorder
        io_rank=3,
        served_compute_ranks=(),
    )
    # a decomposed rank: two halo entries past the owned ones on every dimension
    owned_counts = {dims.CellDim: 5, dims.EdgeDim: 7, dims.VertexDim: 3}
    decomposition_info = decomposition_defs.DecompositionInfo()
    for dim, owned in owned_counts.items():
        owner_mask = np.asarray([True] * owned + [False, False])
        decomposition_info.set_dimension(
            dim,
            np.arange(owned + 2, dtype=np.int64),
            owner_mask,
            np.where(
                owner_mask,
                decomposition_defs.DecompositionFlag.OWNED.value,
                decomposition_defs.DecompositionFlag.FIRST_HALO_LEVEL.value,
            ),
        )
    io_config = common_io.IOConfig(
        output_path=tmp_path,
        field_groups=[common_io.FieldGroupIOConfig(basename="a", variables=["air_density"])],
        io_ranks_per_node=1,
    )

    io_monitor = driver_io.create_io_monitor(
        io_config=io_config,
        grid_file_path=tmp_path / "grid.nc",
        grid=grid,
        vertical_grid=None,  # type: ignore[arg-type] # not used by the recorder
        dtime=datetime.timedelta(seconds=1),
        process_props=decomposition_defs.SingleNodeProcessProperties(),
        decomposition_info=decomposition_info,
        io_ranks=io_ranks,
    )
    assert isinstance(io_monitor, _RecordingClient)
    assert recorded["config"] is io_config
    assert recorded["grid_id"] == uuid.UUID(grid.id)
    # the IO rank gets the undecomposed sizes (the owned entries over the compute
    # ranks), not this rank's halo-including grid sizes, which differ between ranks
    assert recorded["horizontal_size"] == base.HorizontalGridSize(
        num_cells=5, num_edges=7, num_vertices=3
    )
    assert recorded["horizontal_size"] != grid.config.horizontal_config
    assert recorded["intercomm"] is intercomm
    assert recorded["decomposition_info"] is decomposition_info
    # the client ships to the IO rank the split assigned to this compute rank
    assert recorded["io_rank"] == 3

    with pytest.raises(ValueError, match="requires 'decomposition_info'"):
        driver_io.create_io_monitor(
            io_config=io_config,
            grid_file_path=tmp_path / "grid.nc",
            grid=grid,
            vertical_grid=None,  # type: ignore[arg-type] # not used by the recorder
            dtime=datetime.timedelta(seconds=1),
            process_props=decomposition_defs.SingleNodeProcessProperties(),
            decomposition_info=None,
            io_ranks=io_ranks,
        )
    # the split and the configured rank count must agree, in both directions
    for mismatched_config, mismatched_split in (
        (dataclasses.replace(io_config, io_ranks_per_node=0), io_ranks),
        (io_config, None),
    ):
        with pytest.raises(ValueError, match="must agree"):
            driver_io.create_io_monitor(
                io_config=mismatched_config,
                grid_file_path=tmp_path / "grid.nc",
                grid=grid,
                vertical_grid=None,  # type: ignore[arg-type] # not used by the recorder
                dtime=datetime.timedelta(seconds=1),
                process_props=decomposition_defs.SingleNodeProcessProperties(),
                decomposition_info=decomposition_info,
                io_ranks=mismatched_split,
            )


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
    physics_diagnostics: Iterable[physics_state.DiagnosticOutput] | None = None,
) -> driver_io.OutputStateAssembler:
    return driver_io.OutputStateAssembler(
        io_config=io_config,
        grid=grid,
        backend=None,
        static_field_factories=StubStaticFields(grid),  # type: ignore[arg-type]  # stand-in serving the static fields
        tracer_config=tracer_config,
        physics_diagnostics=physics_diagnostics,
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


def test_available_variables_include_the_physics_diagnostic_outputs(grid: base.Grid) -> None:
    assembler = _assembler(
        grid,
        _io_config(("precipitation", ["rainfall_flux"])),
        physics_diagnostics=allocated_physics_outputs(grid, rainfall_outputs()),
    )
    assert assembler.available_variables == {*ALL_OUTPUT_VARIABLES, "rainfall_flux"}


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
        physics_outputs=driver_output_sources.physics_diagnostic_outputs(
            allocated_physics_outputs(grid, {"muphys": {"rho": clashing}})
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
    prognostic_state = uniform_prognostic_state(grid)
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
        physics_diagnostics=allocated_physics_outputs(grid, rainfall_outputs()),
    )

    diagnostic_state = make_diagnostic_state(grid)
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

    monkeypatch.setattr(driver_diagnostics, "DiagnosticsComputer", _Unexpected)
    _assembler(grid, _io_config(("prognostics", ["air_density", "normal_velocity"])))
