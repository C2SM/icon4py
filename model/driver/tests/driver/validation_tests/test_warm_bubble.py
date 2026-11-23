import datetime
import pathlib
from typing import Final

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest
import xarray as xr
from scipy import interpolate, signal

from icon4py.model.common import model_backends, time
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.io import io as common_io
from icon4py.model.common.states import data as state_data
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import definitions as test_defs, grid_utils

from ..fixtures import *  # noqa: F403


_AIR_DENSITY: Final = state_data.PROGNOSTIC_CF_ATTRIBUTES["air_density"].standard_name
_WATER_SPECIES: Final = tuple(state_data.COMMON_TRACER_CF_ATTRIBUTES)
_CONDENSATES: Final = tuple(name for name in _WATER_SPECIES if name != "qv")
_SURFACE_WATER_FLUXES: Final = tuple(
    name
    for name in state_data.PRECIPITATION_CF_ATTRIBUTES
    if name not in ("precipitation_flux", "precipitation_energy_flux")
)

_ATMOSPHERIC_STATE_GROUP: Final = "atmospheric_state"
_PRECIPITATION_GROUP: Final = "precipitation"
_ATMOSPHERIC_STATE_OUTPUT_INTERVAL: Final = datetime.timedelta(minutes=2)
_PRECIPITATION_OUTPUT_INTERVAL: Final = time.NumTimeSteps(1)

_MASS_RTOL: Final = 1e-12
_WATER_RTOL: Final = 1e-12

_SPLIT_TIME: Final = datetime.timedelta(minutes=100)
_MIN_PROMINENCE: Final = 0.2


def _validation_io_config(output_path: pathlib.Path) -> common_io.IOConfig:
    """Just the fields the checks need: the density and the tracers on a coarse
    schedule, and the surface precipitation every step so its time integral is exact."""
    return common_io.IOConfig(
        output_path=output_path,
        field_groups=[
            common_io.FieldGroupIOConfig(
                basename=_ATMOSPHERIC_STATE_GROUP,
                variables=[_AIR_DENSITY, *_WATER_SPECIES],
                output_interval=_ATMOSPHERIC_STATE_OUTPUT_INTERVAL,
                timesteps_per_file=0,
                backend=common_io.OutputBackend.NETCDF,
                mode=common_io.OutputMode.GATHER,
            ),
            common_io.FieldGroupIOConfig(
                basename=_PRECIPITATION_GROUP,
                variables=list(_SURFACE_WATER_FLUXES),
                output_interval=_PRECIPITATION_OUTPUT_INTERVAL,
                timesteps_per_file=0,
                backend=common_io.OutputBackend.NETCDF,
                mode=common_io.OutputMode.GATHER,
            ),
        ],
    )


def _open_group(output_path: pathlib.Path, basename: str) -> xr.Dataset:
    files = sorted(output_path.rglob(f"{basename}_*.nc"))
    assert len(files) == 1, f"expected one '{basename}' output file, found {files}"
    return xr.open_dataset(files[0], decode_times=False)


def _open_grid(output_path: pathlib.Path) -> xr.Dataset:
    files = sorted(output_path.rglob("*_ugrid.nc"))
    assert len(files) == 1, f"expected one UGRID file, found {files}"
    return xr.open_dataset(files[0])


def _layer_thickness(data: xr.Dataset) -> np.ndarray:
    """Thickness (m) of the full levels from the half-level heights (top to bottom)."""
    h = data["height"].values
    return h[:-1] - h[1:]


def _surface_flux(flux: xr.DataArray) -> xr.DataArray:
    return flux.isel(level=-1) if "level" in flux.dims else flux


def _cumulative_surface_precipitation(
    precipitation: xr.Dataset, times: np.ndarray, dtime: float, weights: np.ndarray
) -> np.ndarray:
    flux = sum(_surface_flux(precipitation[name]).values for name in _SURFACE_WATER_FLUXES)
    mean_flux_time_series = np.sum(flux * weights[None, :], axis=-1)
    precip_times = precipitation["time"].values
    np.testing.assert_allclose(
        np.diff(precip_times), dtime, err_msg="precipitation is not output at every time step"
    )
    cumulative = np.concatenate(([0.0], np.cumsum(mean_flux_time_series[1:] * dtime)))
    return np.interp(times, precip_times, cumulative)


def _y_rows(y: np.ndarray, domain_height: float) -> tuple[np.ndarray, np.ndarray]:
    """sorted the y-center of the cells, rounded to 3 decimal places."""
    y_center, y_index = np.unique(np.round(y, 3), return_inverse=True)
    assert y_center[-1] - y_center[0] < domain_height
    return y_center, y_index


def _x_mean(data: np.ndarray, y_index: np.ndarray, area: np.ndarray, nrows: int) -> np.ndarray:
    """Area weighted mean of data over every row of the data (i.e. over x)."""
    return np.bincount(y_index, weights=area * data, minlength=nrows) / np.bincount(
        y_index, weights=area, minlength=nrows
    )


def _profile_maxima(
    y_center: np.ndarray, data: np.ndarray, domain_height: float, min_prominence: float
) -> np.ndarray:
    """find the y-coordinate of maxima of a data, using cubic spline interpolation to get a smooth profile."""
    top = float(data.max())
    if top <= 0.0:
        return np.empty(0)
    nfine = y_center.size * 3 # 2048
    spline = interpolate.CubicSpline(
        np.append(y_center, y_center[0] + domain_height), np.append(data, data[0]), bc_type="periodic"
        # y_center, data, bc_type="natural"
    )
    y_fine = y_center[0] + np.arange(nfine) * (domain_height / nfine)
    value_fine = spline(y_fine)
    peaks, _ = signal.find_peaks(np.tile(value_fine, 3), prominence=min_prominence * top)
    peaks = peaks[(peaks >= nfine) & (peaks < 2 * nfine)] - nfine
    return y_fine[peaks]
    

def _check_mass_conservation(
    atmospheric_state: xr.Dataset, precipitation: xr.Dataset, weights: np.ndarray, dz: np.ndarray, dtime: float
) -> None:
    layer_mass = atmospheric_state[_AIR_DENSITY].values * dz[None, :, None]
    q_total = sum(atmospheric_state[name].values for name in _WATER_SPECIES)
    air_mass = np.sum(np.sum(layer_mass, axis=1) * weights[None, :], axis=-1)
    water_mass = np.sum(np.sum(q_total * layer_mass, axis=1) * weights[None, :], axis=-1)
    times = atmospheric_state["time"].values
    precipitated_mass = _cumulative_surface_precipitation(precipitation, times, dtime, weights)
    assert precipitated_mass[-1] > 0.0, "no precipitation reached the ground, the budget is trivial"

    np.testing.assert_allclose(
        air_mass,
        air_mass[0],
        rtol=_MASS_RTOL,
        atol=0.0,
        err_msg="total air mass is not conserved",
    )
    np.testing.assert_allclose(
        water_mass + precipitated_mass,
        water_mass[0],
        rtol=_WATER_RTOL,
        atol=0.0,
        err_msg="total water mass plus the surface precipitation is not conserved",
    )


def _check_storm_split(atmospheric_state: xr.Dataset, grid: xr.Dataset, area: np.ndarray, dz: np.ndarray) -> None:
    condensate = sum(atmospheric_state[name].values for name in _CONDENSATES)
    water_path = np.sum(atmospheric_state[_AIR_DENSITY].values * condensate * dz[None, :, None], axis=1)

    domain_height = float(grid.attrs["domain_height"])
    y_center, y_index = _y_rows(grid["cell_circumcenter_cartesian_y"].values, domain_height)
    times = atmospheric_state["time"].values
    elapsed = times - times[0]
    tested = np.flatnonzero(elapsed >= _SPLIT_TIME.total_seconds() - 1e-6)
    assert tested.size > 0, f"the run ends before {_SPLIT_TIME}, no time to test the split at"

    maxima = {
        float(elapsed[it] / 60.0): _profile_maxima(
            y_center=y_center,
            data=_x_mean(
                data=water_path[it],
                y_index=y_index, 
                area=area,
                nrows=y_center.size
            ),
            domain_height=domain_height,
            min_prominence=_MIN_PROMINENCE,
        )
        for it in tested
    }
    not_split = {
        t: [round(float(y) * 1e-3, 1) for y in ys] for t, ys in maxima.items() if ys.size != 2
    }
    assert not not_split, (
        f"the x-mean condensed water path does not have two maxima in y at every time after "
        f"{_SPLIT_TIME}; maxima (km) at the failing times (min): {not_split}"
    )


@pytest.mark.level("validation")
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize("experiment_case", ["warm_bubble"])
def test_warm_bubble(
    *,
    experiment_case: str,
    experiment_config: driver_config.ExperimentConfig,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    output_path = tmp_path / "output"
    config = experiment_config.with_overrides(io=_validation_io_config(output_path))
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_utils._download_grid_file(test_defs.Grids.TORUS_20X24_5000M),
        vertical_grid_config=config.vertical_grid,
        allocator=model_backends.get_allocator(backend),
        process_props=process_props,
    )
    driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )
    if process_props.rank != 0:
        return

    atmospheric_state_data = _open_group(output_path, _ATMOSPHERIC_STATE_GROUP)
    precipitation_data = _open_group(output_path, _PRECIPITATION_GROUP)
    grid_data = _open_grid(output_path)
    for name in (_AIR_DENSITY, *_WATER_SPECIES):
        assert np.isfinite(atmospheric_state_data[name].values).all(), f"model crashed, {name} is not finite"
    area = grid_data["cell_area"].values
    height = atmospheric_state_data["height"].values
    dz = height[:-1] - height[1:]
    weights = area / area.sum()
    _check_mass_conservation(atmospheric_state_data, precipitation_data, weights, dz, config.driver.dtime.total_seconds())
    _check_storm_split(atmospheric_state_data, grid_data, area, dz)
