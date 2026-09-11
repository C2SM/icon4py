# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the driver tests."""

import dataclasses
import math
import pathlib
import time as wall_time
from typing import Final

import gt4py.next.typing as gtx_typing
import netCDF4 as nc
import numpy as np

from icon4py.model.common import dimension as dims, model_backends, time
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import geometry_attributes as geom_attr, gridfile
from icon4py.model.common.initial_condition.analytical import moving_cylinder
from icon4py.model.driver import config as driver_config, driver, driver_io, driver_utils
from icon4py.model.testing import config as test_config


def read_qv_frames(output_dir: pathlib.Path) -> np.ndarray:
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


# --- Jocksch's moving-cylinder experiment (integration_tests/test_jocksch_cylinder*.py) ---

CYLINDER_EXPERIMENT_CONFIG: Final = test_config.EXPERIMENT_CONFIG_PATH / "jocksch_cylinder.yaml"
#: both torus grid files of the experiment are 20 x 22 with 5 km edges
CYLINDER_GRID_SIZES: Final = {dims.CellDim: 880, dims.EdgeDim: 1320, dims.VertexDim: 440}
CYLINDER_RADIUS: Final = 25000.0
CYLINDER_WIND_SPEED: Final = 1.0
#: wind direction, counter-clockwise from the x axis [degrees]
CYLINDER_WIND_ANGLE: Final = 0.0
CYLINDER_CFL: Final = 0.2
CYLINDER_N_TIME_STEPS: Final = 100
#: total tracer mass sum(qv * airmass * cell_area) is conserved to round-off
MASS_CONSERVATION_RTOL: Final = 1e-12


@dataclasses.dataclass(frozen=True)
class CylinderRun:
    """One period of the cylinder: the final state's error measures and what the gates need."""

    cell_x: np.ndarray
    cell_y: np.ndarray
    cell_area: np.ndarray
    edge_normal_x: np.ndarray
    edge_length: float
    dtime_seconds: float
    num_levels: int
    elapsed_wall_time: float
    #: (time, cell, level); frame 0 is the initial state
    qv_frames: np.ndarray
    #: the cylinder sampled at the cell centres (frame 0) and the exact final state
    cylinder: np.ndarray
    reference: np.ndarray
    #: final state minus the exact one, on one level
    error: np.ndarray
    #: Jocksch's neighbour-pair sum and the same over all pairs (neighbour_pair_error_sums)
    jocksch_measure: float
    all_pairs_measure: float
    sum_squared_error: float
    overshoot_final: float
    undershoot_final: float
    overshoot_run: float
    undershoot_run: float
    relative_mass_change: float


def neighbour_pair_error_sums(
    *,
    error: np.ndarray,
    c2e2c: np.ndarray,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    edge_length: float,
) -> tuple[float, float]:
    """(Jocksch's measure, the same over all pairs) of sum_(i,j) (e_i^2 + e_j^2).

    Each unordered neighbour pair is counted once. Jocksch's loop skips the pairs whose
    raw centre distance is not below the edge length, i.e. those across the periodic
    boundary; the all-pairs sum is 3 * sum(e^2) on a torus.
    """
    cell = np.repeat(np.arange(c2e2c.shape[0]), c2e2c.shape[1])
    neighbour = c2e2c.ravel()
    once = neighbour > cell
    pair_sum = error[cell] ** 2 + error[neighbour] ** 2
    distance = np.hypot(cell_x[cell] - cell_x[neighbour], cell_y[cell] - cell_y[neighbour])
    return (
        float(pair_sum[once & (distance < edge_length)].sum()),
        float(pair_sum[once].sum()),
    )


def run_cylinder_one_period(
    *,
    grid_file: pathlib.Path,
    cylinder_center: tuple[float | None, float | None],
    tracer_advection: dict,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> CylinderRun:
    """Carry the cylinder once around the torus with the driver and measure the final error.

    The experiment of CYLINDER_EXPERIMENT_CONFIG on the torus of ``grid_file``: the cylinder
    (radius CYLINDER_RADIUS at ``cylinder_center``, None is the domain centre) in a uniform
    CYLINDER_WIND_SPEED wind; with dt = CYLINDER_CFL * edge_length the CYLINDER_N_TIME_STEPS
    steps are exactly one period in x, so the exact final state is the initial field.
    ``tracer_advection`` are the overrides of the tracer-advection configuration (scheme,
    limiter, weight set). Checks the plumbing every case shares: the grid sizes, a fully
    periodic torus (no skip values), unit air mass, one frame per step with identical
    columns, frame 0 the sampled cylinder, mass conservation to MASS_CONSERVATION_RTOL and the
    all-pairs identity of neighbour_pair_error_sums.
    """
    allocator = model_backends.get_allocator(backend)

    ic_config = moving_cylinder.MovingCylinderConfig(
        center_x=cylinder_center[0],
        center_y=cylinder_center[1],
        radius=CYLINDER_RADIUS,
        wind_speed=CYLINDER_WIND_SPEED,
        wind_angle=CYLINDER_WIND_ANGLE,
    )
    experiment_config = config_io.read_yaml_str(
        CYLINDER_EXPERIMENT_CONFIG.read_text(), driver_config.ExperimentConfig
    ).with_overrides(initial_condition={"config": ic_config})

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file,
        vertical_grid_config=experiment_config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    grid = grid_manager.grid
    for dim, expected_size in CYLINDER_GRID_SIZES.items():
        assert grid.size[dim] == expected_size, f"{dim.value}: {grid.size[dim]} != {expected_size}"
    domain_length = grid.grid_params.domain_length
    domain_height = grid.grid_params.domain_height
    assert domain_length is not None and domain_height is not None
    # a fully periodic torus grid has no skip (invalid) indices
    for offset in (dims.C2E2C, dims.E2C):
        table = grid.get_connectivity(offset).asnumpy()
        assert (table >= 0).all(), f"skip values in {offset.value}: grid is not fully periodic"

    edge_length = float(
        grid_manager.geometry_fields[gridfile.GeometryName.EDGE_LENGTH].asnumpy().mean()
    )
    dtime_seconds = CYLINDER_CFL * edge_length / CYLINDER_WIND_SPEED
    experiment_config = experiment_config.with_overrides(
        driver={
            "output_path": tmp_path / "driver_output",
            "dtime": time.RelativeTime(seconds=dtime_seconds),
            "end_of_simulation": time.NumTimeSteps(CYLINDER_N_TIME_STEPS),
        },
        tracer_advection=tracer_advection,
    )

    start = wall_time.perf_counter()
    ds, icon4py_driver = driver.run_driver(
        config=experiment_config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )
    elapsed_wall_time = wall_time.perf_counter() - start

    geometry = icon4py_driver.static_field_factories.geometry
    cell_x = geometry.get(geom_attr.CELL_CENTER_X).asnumpy()
    cell_y = geometry.get(geom_attr.CELL_CENTER_Y).asnumpy()
    cell_area = geometry.get(geom_attr.CELL_AREA).asnumpy()
    edge_normal_x = geometry.get(geom_attr.EDGE_NORMAL_U).asnumpy()
    assert ds.tracer_advection_diagnostic is not None
    airmass = ds.tracer_advection_diagnostic.airmass_now.asnumpy()
    np.testing.assert_allclose(airmass, 1.0, rtol=1e-14)

    # frame 0 is the initial state, one frame per step afterwards
    qv_frames = read_qv_frames(tmp_path)
    num_levels = experiment_config.vertical_grid.num_levels
    assert qv_frames.shape == (CYLINDER_N_TIME_STEPS + 1, grid.num_cells, num_levels)
    assert np.isfinite(qv_frames).all()
    # the columns are identical, so the errors are taken on one level
    np.testing.assert_array_equal(qv_frames, np.broadcast_to(qv_frames[:, :, :1], qv_frames.shape))

    # the initial frame is the cylinder sampled at the cell centres
    cylinder = moving_cylinder.sample_cylinder(
        config=ic_config,
        cell_center_x=cell_x,
        cell_center_y=cell_y,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    np.testing.assert_array_equal(qv_frames[0, :, 0], cylinder)

    # the exact solution is the cylinder displaced by the wind; for the default
    # parameters that is exactly one period in x, i.e. the initial cylinder again
    u, v = moving_cylinder.compute_wind_components(ic_config)
    elapsed_time = CYLINDER_N_TIME_STEPS * dtime_seconds
    center_x, center_y = moving_cylinder.compute_cylinder_center(
        ic_config, domain_length, domain_height
    )
    displaced_config = moving_cylinder.MovingCylinderConfig(
        center_x=(center_x + u * elapsed_time) % domain_length,
        center_y=(center_y + v * elapsed_time) % domain_height,
        radius=CYLINDER_RADIUS,
    )
    reference = moving_cylinder.sample_cylinder(
        config=displaced_config,
        cell_center_x=cell_x,
        cell_center_y=cell_y,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    one_period = math.isclose(u * elapsed_time % domain_length, 0.0, abs_tol=1e-6) and (
        math.isclose(v * elapsed_time % domain_height, 0.0, abs_tol=1e-6)
    )
    if one_period:
        np.testing.assert_array_equal(reference, cylinder)
        reference = cylinder

    # total tracer mass sum(qv * airmass * cell_area) is conserved
    mass = np.einsum("tck,ck,c->t", qv_frames, airmass, cell_area)
    assert mass[0] > 0.0, "initial tracer mass is zero: prescription plumbing is missing"
    np.testing.assert_allclose(mass, mass[0], rtol=MASS_CONSERVATION_RTOL)

    q_final = qv_frames[-1, :, 0]
    error = q_final - reference
    sum_squared_error = float(np.sum(error**2))
    jocksch_measure, all_pairs_measure = neighbour_pair_error_sums(
        error=error,
        c2e2c=grid.get_connectivity(dims.C2E2C).asnumpy(),
        cell_x=cell_x,
        cell_y=cell_y,
        edge_length=edge_length,
    )
    np.testing.assert_allclose(all_pairs_measure, 3.0 * sum_squared_error, rtol=1e-12)

    return CylinderRun(
        cell_x=cell_x,
        cell_y=cell_y,
        cell_area=cell_area,
        edge_normal_x=edge_normal_x,
        edge_length=edge_length,
        dtime_seconds=dtime_seconds,
        num_levels=num_levels,
        elapsed_wall_time=elapsed_wall_time,
        qv_frames=qv_frames,
        cylinder=cylinder,
        reference=reference,
        error=error,
        jocksch_measure=jocksch_measure,
        all_pairs_measure=all_pairs_measure,
        sum_squared_error=sum_squared_error,
        overshoot_final=float(q_final.max() - 1.0),
        undershoot_final=float(-q_final.min()),
        overshoot_run=float(qv_frames.max() - 1.0),
        undershoot_run=float(-qv_frames.min()),
        relative_mass_change=float((mass[-1] - mass[0]) / mass[0]),
    )
