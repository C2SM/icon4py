# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tracer-advection-only experiment: a passive tracer disc in a constant wind on a torus.

Dycore and diffusion are off; the 'tracer_blob' initial condition prescribes the mass
fluxes the dycore would normally provide. The disc must translate with the wind, so the
final tracer field is checked against the analytically translated initial disc and the
total tracer mass must be conserved. Runs with a torus grid file only (no serialized
data).
"""

import datetime
import math
import pathlib

import gt4py.next.typing as gtx_typing
import netCDF4 as nc
import numpy as np
import pytest

from icon4py.model.atmosphere.advection import advection as tracer_advection
from icon4py.model.common import dimension as dims, initial_condition, model_backends, time
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import (
    geometry_attributes as geom_attr,
    geometry_config as geometry_configuration,
    vertical as v_grid,
)
from icon4py.model.common.initial_condition.analytical import tracer_blob as tracer_blob_ic
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import tracer_state
from icon4py.model.common.topography import config as topography_config
from icon4py.model.common.topography.analytical import flat_topography as flat_topo
from icon4py.model.standalone_driver import (
    config as driver_config,
    driver_io,
    driver_utils,
    standalone_driver,
)
from icon4py.model.testing import definitions as test_defs, grid_utils

from ..fixtures import *  # noqa: F403


GRID = test_defs.Grids.TORUS_50000x5000
NUM_LEVELS = 10
U0 = 20.0
CFL = 0.3
N_TIME_STEPS = 24


def _translated_disc(
    *,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    config: tracer_blob_ic.TracerBlobConfig,
    domain_length: float,
    domain_height: float,
    elapsed_time: float,
) -> np.ndarray:
    """The initial disc translated by u0 * t along x with torus-periodic wrap."""
    blob_x = config.blob_x if config.blob_x is not None else 0.5 * domain_length
    blob_y = config.blob_y if config.blob_y is not None else 0.5 * domain_height
    radius = (
        config.blob_radius
        if config.blob_radius is not None
        else 0.25 * min(domain_length, domain_height)
    )
    center_x = (blob_x + config.u0 * elapsed_time) % domain_length
    dx = (cell_x - center_x + 0.5 * domain_length) % domain_length - 0.5 * domain_length
    dy = (cell_y - blob_y + 0.5 * domain_height) % domain_height - 0.5 * domain_height
    # squared form (no hypot/sqrt) to match the 'tracer_blob' IC bit-for-bit at the disc boundary
    return np.where(np.less_equal(dx**2 + dy**2, radius**2), config.blob_amplitude, 0.0)


def _read_qv_frames(output_dir: pathlib.Path) -> np.ndarray:
    """qv from the driver output as (time, cell, level)."""
    output_files = sorted(output_dir.rglob(f"{driver_io.DEFAULT_OUTPUT_FILENAME}_*.nc"))
    assert output_files, f"no output file under {output_dir}"
    frames = []
    for output_file in output_files:
        with nc.Dataset(output_file) as ds:
            assert "qv" in ds.variables, "qv missing from driver output"
            var = ds.variables["qv"]
            axes = [var.dimensions.index(name) for name in ("time", "cell", "level")]
            frames.append(np.transpose(np.asarray(var[:]), axes))
    return np.concatenate(frames, axis=0)


# Observed relative L2 errors vs the translated disc after 24 steps (CFL~0.3, blob
# defaults, torus_50000mx5000m_res500m, gtfn_cpu): miura3 WENO (103) 0.297,
# WENO (102) 0.328, plain miura (2) 0.481. The tolerances are frozen at ~1.5x the
# observed values; a non-moving disc gives ~1.4.
@pytest.mark.level("integration")
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "horizontal_advection_type, l2_tolerance",
    [
        (tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER_WENO, 0.50),
        (tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER, 0.72),
        (tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER_WENO, 0.45),
    ],
)
def test_tracer_blob_translation(
    horizontal_advection_type: tracer_advection.HorizontalAdvectionType,
    l2_tolerance: float,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)
    grid_file_path = grid_utils._download_grid_file(GRID)

    vertical_grid_config = v_grid.VerticalGridConfig(num_levels=NUM_LEVELS)
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=vertical_grid_config,
        allocator=allocator,
        process_props=process_props,
    )
    grid = grid_manager.grid
    domain_length = grid.grid_params.domain_length
    domain_height = grid.grid_params.domain_height
    assert domain_length is not None and domain_height is not None

    # periodicity precheck: a fully periodic torus grid has no skip (invalid) indices
    for offset in (dims.C2E2C, dims.E2C):
        table = grid.get_connectivity(offset).asnumpy()
        assert (table >= 0).all(), f"skip values in {offset.value}: grid is not fully periodic"

    # equilateral-triangle edge length from the mean cell area; dtime from CFL ~ 0.3
    mean_cell_area = domain_length * domain_height / grid.num_cells
    edge_length = math.sqrt(4.0 * mean_cell_area / math.sqrt(3.0))
    dtime_seconds = CFL * edge_length / U0

    ic_config = tracer_blob_ic.TracerBlobConfig(u0=U0)
    config = driver_config.ExperimentConfig(
        geometry=geometry_configuration.GeometryConfig(),
        metrics=metrics_factory.MetricsConfig(),
        interpolation=interpolation_factory.InterpolationConfig(),
        vertical_grid=vertical_grid_config,
        topography=topography_config.TopographyConfig(config=flat_topo.FlatTopographyConfig()),
        initial_condition=initial_condition.InitialConditionConfig(config=ic_config),
        driver=driver_config.DriverConfig(
            experiment_name="tracer_blob",
            profiling_stats=None,
            dtime=time.RelativeTime(seconds=dtime_seconds),
            start_of_simulation=datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC),
            end_of_simulation=time.NumTimeSteps(N_TIME_STEPS),
            output_path=tmp_path / "driver_output",
            enable_output=True,
        ),
        nonhydrostatic=None,
        diffusion=None,
        tracer_config=tracer_state.TracerConfig(qv=True),
        tracer_advection=tracer_advection.AdvectionConfig(
            horizontal_advection_type=horizontal_advection_type,
            horizontal_advection_limiter=tracer_advection.HorizontalAdvectionLimiter.NO_LIMITER,
            vertical_advection_type=tracer_advection.VerticalAdvectionType.NO_ADVECTION,
            vertical_advection_limiter=tracer_advection.VerticalAdvectionLimiter.NO_LIMITER,
        ),
    )

    ds, icon4py_driver = standalone_driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )

    geometry = icon4py_driver.static_field_factories.geometry
    cell_x = geometry.get(geom_attr.CELL_CENTER_X).asnumpy()
    cell_y = geometry.get(geom_attr.CELL_CENTER_Y).asnumpy()
    cell_area = geometry.get(geom_attr.CELL_AREA).asnumpy()

    assert ds.tracer_advection_diagnostic is not None
    airmass = ds.tracer_advection_diagnostic.airmass_now.asnumpy()

    # frame 0 is the initial state, one frame per step afterwards
    qv_frames = _read_qv_frames(tmp_path)
    assert qv_frames.shape[0] == N_TIME_STEPS + 1

    # the initial frame must be the sampled disc (nothing moved yet)
    disc_now = _translated_disc(
        cell_x=cell_x,
        cell_y=cell_y,
        config=ic_config,
        domain_length=domain_length,
        domain_height=domain_height,
        elapsed_time=0.0,
    )
    np.testing.assert_array_equal(
        qv_frames[0], np.broadcast_to(disc_now[:, np.newaxis], qv_frames[0].shape)
    )

    # total tracer mass sum(qv * airmass * cell_area) is conserved
    mass = np.einsum("tck,ck,c->t", qv_frames, airmass, cell_area)
    assert mass[0] > 0.0, "initial tracer mass is zero: prescription plumbing is missing"
    np.testing.assert_allclose(mass, mass[0], rtol=1e-12)

    # the disc must have translated by u0 * t (torus-periodic); loose L2 check.
    # A tracer that does not move at all gives a relative L2 error of ~1.4.
    elapsed_time = N_TIME_STEPS * dtime_seconds
    disc_translated = _translated_disc(
        cell_x=cell_x,
        cell_y=cell_y,
        config=ic_config,
        domain_length=domain_length,
        domain_height=domain_height,
        elapsed_time=elapsed_time,
    )
    reference = np.broadcast_to(disc_translated[:, np.newaxis], qv_frames[-1].shape)
    l2_error = np.linalg.norm(qv_frames[-1] - reference) / np.linalg.norm(reference)
    print(f"{horizontal_advection_type.name}: relative L2 error vs translated disc = {l2_error}")
    assert l2_error < l2_tolerance
