# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Jocksch's moving-cylinder experiment for the FFSL-WENO schemes on the shared torus.

A tracer cylinder (radius 25 km) is carried once around a periodic 20 x 22 torus with
5 km edges by a uniform 1 m/s wind: with dt = CFL * edge_length = 1000 s, 100 steps are
exactly one period in x, so the exact solution is the initial field and the error is
q(100) - q(0). The grid file is the one the Fortran reference run uses, so the numbers
printed here can be set next to Table 2 of the paper. Only sanity is asserted for now;
the error table is printed (run with '-s').

The error measures follow the live block in Jocksch's 'mo_nh_stepping.f90': his measure
sums (e_i^2 + e_j^2) over the unordered neighbour-cell pairs (i, j) whose raw (non-
periodic) centre distance is below the edge length, which drops the pairs across the
periodic boundary; with those pairs it would be exactly 3 * sum(e^2) on a torus.
"""

import math
import pathlib
import time as wall_time
from typing import Final

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection import tracer_advection
from icon4py.model.common import dimension as dims, model_backends, time
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import geometry_attributes as geom_attr, gridfile
from icon4py.model.common.initial_condition.analytical import moving_cylinder
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import config as test_config

from .. import utils as test_utils
from ..fixtures import *  # noqa: F403


#: the planar torus shared with the Fortran reference run: 20 x 22, 5 km edges,
#: 880 cells / 1320 edges / 440 vertices, 100 km x 95.26 km
GRID_FILE: Final = pathlib.Path(
    "/capstor/scratch/cscs/cmueller/tracer_advection_port/icon-exclaim/weno_data/grids/"
    "torus_20x22_res5000m.nc"
)
EXPECTED_GRID_SIZES: Final = {dims.CellDim: 880, dims.EdgeDim: 1320, dims.VertexDim: 440}
EXPERIMENT_CONFIG: Final = test_config.EXPERIMENT_CONFIG_PATH / "jocksch_cylinder.yaml"

#: cylinder centre in the grid file's coordinates, None is the domain centre
CYLINDER_CENTER: Final[tuple[float | None, float | None]] = (None, None)
CYLINDER_RADIUS: Final = 25000.0
WIND_SPEED: Final = 1.0
#: wind direction, counter-clockwise from the x axis [degrees]
WIND_ANGLE: Final = 0.0
CFL: Final = 0.2
N_TIME_STEPS: Final = 100
#: total tracer mass sum(qv * airmass * cell_area) is conserved to round-off
_MASS_CONSERVATION_RTOL: Final = 1e-12

_HADV = tracer_advection.HorizontalAdvectionType
_HLIM = tracer_advection.HorizontalAdvectionLimiter

#: Table 2 of the paper, by (scheme, limiter). Measured here (gtfn_cpu), sqrt(sum e^2)
#: reproduces every entry to the printed digits, 4.0240 / 3.8594 / 3.4029 / 3.0586 /
#: 3.3712 (the paper truncates), so that is the paper's normalisation rather than the
#: neighbour-pair sum of his loop. Nothing is asserted against it yet: the Fortran
#: reference run on this grid file is what settles the tolerance.
PAPER_TABLE_2: Final[dict[tuple[_HADV, _HLIM], float]] = {
    (_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER): 4.023,
    (_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER): 3.859,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER): 3.402,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER): 3.058,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC): 3.371,
}


def _neighbour_pair_error_sums(
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


@pytest.mark.level("integration")
@pytest.mark.embedded_remap_error
@pytest.mark.skipif(not GRID_FILE.exists(), reason=f"shared grid file {GRID_FILE} not found")
@pytest.mark.parametrize(
    "horizontal_advection_type, horizontal_advection_limiter",
    [
        pytest.param(_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER, id="miura"),
        pytest.param(_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER, id="miura3"),
        pytest.param(_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER, id="miura_weno"),
        pytest.param(_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, id="miura3_weno"),
        pytest.param(_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC, id="miura-monotonic"),
        pytest.param(_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC, id="miura3-monotonic"),
        pytest.param(_HADV.LINEAR_2ND_ORDER, _HLIM.POSITIVE_DEFINITE, id="miura-positive_definite"),
        pytest.param(
            _HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE, id="miura3-positive_definite"
        ),
    ],
)
def test_jocksch_cylinder_one_period(
    horizontal_advection_type: tracer_advection.HorizontalAdvectionType,
    horizontal_advection_limiter: tracer_advection.HorizontalAdvectionLimiter,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)

    ic_config = moving_cylinder.MovingCylinderConfig(
        center_x=CYLINDER_CENTER[0],
        center_y=CYLINDER_CENTER[1],
        radius=CYLINDER_RADIUS,
        wind_speed=WIND_SPEED,
        wind_angle=WIND_ANGLE,
    )
    experiment_config = config_io.read_yaml_str(
        EXPERIMENT_CONFIG.read_text(), driver_config.ExperimentConfig
    ).with_overrides(initial_condition={"config": ic_config})

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=GRID_FILE,
        vertical_grid_config=experiment_config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    grid = grid_manager.grid
    for dim, expected_size in EXPECTED_GRID_SIZES.items():
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
    dtime_seconds = CFL * edge_length / WIND_SPEED
    experiment_config = experiment_config.with_overrides(
        driver={
            "output_path": tmp_path / "driver_output",
            "dtime": time.RelativeTime(seconds=dtime_seconds),
            "end_of_simulation": time.NumTimeSteps(N_TIME_STEPS),
        },
        tracer_advection={
            "horizontal_advection_type": horizontal_advection_type,
            "horizontal_advection_limiter": horizontal_advection_limiter,
        },
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
    assert ds.tracer_advection_diagnostic is not None
    airmass = ds.tracer_advection_diagnostic.airmass_now.asnumpy()
    np.testing.assert_allclose(airmass, 1.0, rtol=1e-14)

    # frame 0 is the initial state, one frame per step afterwards
    qv_frames = test_utils.read_qv_frames(tmp_path)
    num_levels = experiment_config.vertical_grid.num_levels
    assert qv_frames.shape == (N_TIME_STEPS + 1, grid.num_cells, num_levels)
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
    elapsed_time = N_TIME_STEPS * dtime_seconds
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

    # total tracer mass sum(qv * airmass * cell_area) is conserved
    mass = np.einsum("tck,ck,c->t", qv_frames, airmass, cell_area)
    assert mass[0] > 0.0, "initial tracer mass is zero: prescription plumbing is missing"
    relative_mass_change = (mass[-1] - mass[0]) / mass[0]
    np.testing.assert_allclose(mass, mass[0], rtol=_MASS_CONSERVATION_RTOL)

    q_final = qv_frames[-1, :, 0]
    error = q_final - reference
    sum_squared_error = float(np.sum(error**2))
    jocksch_measure, all_pairs_measure = _neighbour_pair_error_sums(
        error=error,
        c2e2c=grid.get_connectivity(dims.C2E2C).asnumpy(),
        cell_x=cell_x,
        cell_y=cell_y,
        edge_length=edge_length,
    )
    np.testing.assert_allclose(all_pairs_measure, 3.0 * sum_squared_error, rtol=1e-12)
    overshoot_final = float(q_final.max() - 1.0)
    undershoot_final = float(-q_final.min())
    overshoot_run = float(qv_frames.max() - 1.0)
    undershoot_run = float(-qv_frames.min())

    paper_value = PAPER_TABLE_2.get((horizontal_advection_type, horizontal_advection_limiter))
    print(
        f"\n{horizontal_advection_type.name} ({horizontal_advection_type.value}) + "
        f"{horizontal_advection_limiter.name} ({horizontal_advection_limiter.value}): "
        f"{N_TIME_STEPS} steps of dt = {dtime_seconds} s, "
        f"{num_levels} level(s), wall time {elapsed_wall_time:.1f} s\n"
        f"  Jocksch measure (pairs within an edge length)  = {jocksch_measure:.6f}"
        f"  sqrt = {math.sqrt(jocksch_measure):.6f}\n"
        f"  all neighbour pairs (= 3 sum e^2)              = {all_pairs_measure:.6f}"
        f"  sqrt = {math.sqrt(all_pairs_measure):.6f}\n"
        f"  sum e^2                                        = {sum_squared_error:.6f}"
        f"  sqrt = {math.sqrt(sum_squared_error):.6f}\n"
        f"  overshoot (max q - 1) final / run              = {overshoot_final:.6e} / "
        f"{overshoot_run:.6e}\n"
        f"  undershoot (-min q) final / run                = {undershoot_final:.6e} / "
        f"{undershoot_run:.6e}\n"
        f"  relative mass change                           = {relative_mass_change:.6e}\n"
        f"  paper Table 2                                  = {paper_value}"
    )
