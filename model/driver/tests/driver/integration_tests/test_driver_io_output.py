# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for single-node output from the driver.

Run the Jablonowski-Williamson testcase for a step or two with output enabled and assert
that valid CF/UGRID NetCDF files (or zarr stores) are produced: the built-in default
group, configured field groups on their own schedules, and the precipitation fluxes of a
moist run with microphysics. They exercise the full ``store -> file`` path, not just the
in-memory bridge.

Being datatests, they require the JW grid and experiment configuration.
"""

import pathlib

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest
import xarray as xr

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import config as muphys_config
from icon4py.model.common import model_backends, time
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.io import io as common_io
from icon4py.model.common.states import tracer_states
from icon4py.model.driver import config as driver_config, driver, driver_io, driver_utils
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs, grid_utils

from ..fixtures import *  # noqa: F403


def _find_one(directory: pathlib.Path, pattern: str) -> pathlib.Path:
    matches = sorted(directory.rglob(pattern))
    assert matches, f"no file matching {pattern!r} under {directory}"
    return matches[0]


def _open_output(path: pathlib.Path, output_backend: common_io.OutputBackend) -> xr.Dataset:
    match output_backend:
        case common_io.OutputBackend.NETCDF:
            return xr.open_dataset(path, decode_times=False)
        case common_io.OutputBackend.ZARR:
            return xr.open_zarr(path, decode_times=False)


def _run_driver(
    experiment_description: test_defs.ExperimentDescription,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    config: driver_config.ExperimentConfig,
) -> None:
    allocator = model_backends.get_allocator(backend)
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_utils._download_grid_file(experiment_description.grid),
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW])
@pytest.mark.parametrize(
    "output_backend", [common_io.OutputBackend.NETCDF, common_io.OutputBackend.ZARR]
)
def test_driver_writes_output(
    experiment_description: test_defs.ExperimentDescription,
    output_backend: common_io.OutputBackend,
    *,
    download_ser_data: None,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)
    config = driver_config.read_experiment_config_from_yaml(config_file_path / "config.yml")
    config = config.with_overrides(
        driver={"end_of_simulation": time.NumTimeSteps(1)},
        io=driver_io.default_io_config(
            tmp_path / "io_driver_output",
            backend=output_backend,
            # gather is valid for both backends everywhere (distributed netCDF
            # would be rejected on serial netCDF4 installations)
            mode=common_io.OutputMode.GATHER,
        ),
    )

    _run_driver(experiment_description, process_props, backend, config)

    # the UGRID grid file is written on monitor init
    _find_one(tmp_path, "*_ugrid.nc")

    # single data file: all prognostic AND diagnostic fields together. Two time slices:
    # the initial state (always written) plus the one integrated step.
    suffix = common_io.FILE_SUFFIXES[output_backend]
    output_file = _find_one(tmp_path, f"{driver_io.DEFAULT_OUTPUT_BASENAME}_*{suffix}")
    with _open_output(output_file, output_backend) as ds:
        assert ds.attrs["Conventions"] == "CF-1.7"
        for name in driver_io.DEFAULT_OUTPUT_VARIABLES:
            assert name in ds.variables, f"{name} missing from output"
            var = ds[name]
            assert var.dims[0] == "time"
            assert var.shape[0] == 2
        # vertical placement: w on half (interface) levels, the rest on full levels
        assert "half_level" in ds["upward_air_velocity"].dims
        assert "edge" in ds["normal_velocity"].dims
        # diagnostics live on cells/full levels
        assert "cell" in ds["temperature"].dims
        # a purely horizontal (surface) diagnostic
        assert ds["surface_pressure"].dims == ("time", "cell")
        assert ds.sizes["time"] == 2


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW])
def test_driver_writes_configured_field_groups(
    experiment_description: test_defs.ExperimentDescription,
    *,
    download_ser_data: None,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    """Two field groups on their own schedules, each in its own file series."""
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)
    config = driver_config.read_experiment_config_from_yaml(config_file_path / "config.yml")
    config = config.with_overrides(
        driver={"end_of_simulation": time.NumTimeSteps(2)},
        io=common_io.IOConfig(
            output_path=tmp_path / "io_driver_output",
            field_groups=[
                common_io.FieldGroupIOConfig(
                    basename="winds",
                    variables=["eastward_wind", "northward_wind", "surface_pressure"],
                    output_interval=time.NumTimeSteps(2),
                    backend=common_io.OutputBackend.NETCDF,
                    mode=common_io.OutputMode.GATHER,
                ),
                common_io.FieldGroupIOConfig(
                    basename="prognostics",
                    variables=list(driver_io.PROGNOSTIC_VARIABLES),
                    backend=common_io.OutputBackend.NETCDF,
                    mode=common_io.OutputMode.GATHER,
                ),
            ],
        ),
    )

    _run_driver(experiment_description, process_props, backend, config)

    # the built-in default group is not written when field groups are configured
    assert not list(tmp_path.rglob(f"{driver_io.DEFAULT_OUTPUT_BASENAME}_*"))
    # the winds every second step: the initial state and step 2
    with _open_output(_find_one(tmp_path, "winds_*.nc"), common_io.OutputBackend.NETCDF) as ds:
        assert {"eastward_wind", "northward_wind", "surface_pressure"} <= set(ds.variables)
        assert not set(driver_io.PROGNOSTIC_VARIABLES) & set(ds.variables)
        assert "temperature" not in ds.variables
        assert ds.sizes["time"] == 2
        # the surface pressure is a purely horizontal field
        assert ds["surface_pressure"].dims == ("time", "cell")
        assert (ds["surface_pressure"].values > 0.0).all()
    # the prognostics every step: the initial state and both steps
    with _open_output(
        _find_one(tmp_path, "prognostics_*.nc"), common_io.OutputBackend.NETCDF
    ) as ds:
        assert set(driver_io.PROGNOSTIC_VARIABLES) <= set(ds.variables)
        assert "eastward_wind" not in ds.variables
        assert ds.sizes["time"] == 3


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.level("integration")
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW])
def test_driver_writes_precipitation_from_muphys(
    experiment_description: test_defs.ExperimentDescription,
    *,
    download_ser_data: None,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    """A moist run with microphysics offers the tracers and the precipitation fluxes."""
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)
    config = driver_config.read_experiment_config_from_yaml(config_file_path / "config.yml")
    config = config.with_overrides(
        driver={"end_of_simulation": time.NumTimeSteps(1)},
        # the physics needs all six moisture species; the JW initial condition fills qv
        tracer_config=tracer_states.TracerConfig.all(),
        muphys=muphys_config.MuphysConfig(),
        io=common_io.IOConfig(
            output_path=tmp_path / "io_driver_output",
            field_groups=[
                common_io.FieldGroupIOConfig(
                    basename="precipitation",
                    variables=["precipitation_flux", "rainfall_flux", "qv", "temperature"],
                    backend=common_io.OutputBackend.NETCDF,
                    mode=common_io.OutputMode.GATHER,
                ),
            ],
        ),
    )

    _run_driver(experiment_description, process_props, backend, config)

    with _open_output(
        _find_one(tmp_path, "precipitation_*.nc"), common_io.OutputBackend.NETCDF
    ) as ds:
        assert ds.sizes["time"] == 2
        for name in ("precipitation_flux", "rainfall_flux", "qv", "temperature"):
            assert name in ds.variables, f"{name} missing from output"
            assert ds[name].dims == ("time", "level", "cell")
            assert np.isfinite(ds[name].values).all()
        assert ds["precipitation_flux"].attrs["standard_name"] == "precipitation_flux"
        assert ds["qv"].attrs["standard_name"] == "specific_humidity"
        # the physics buffers hold zeros before the process has run: the initial slice
        # of the fluxes is zero, the moist initial state is not
        assert not ds["precipitation_flux"].isel(time=0).values.any()
        assert ds["qv"].isel(time=0).values.any()
