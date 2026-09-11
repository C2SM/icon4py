# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Jocksch's moving-cylinder experiment on his own grid: the rows of the paper's Table 2.

The experiment of test_jocksch_cylinder.py (which runs it on the generated torus) on the
grid the numbers of the paper (Jocksch et al., PPAM 2026) come from: his
torus_grid_r4_c200_elen100.nc, 880 cells / 1320 edges / 440 vertices, 5 km edges, centred
coordinates (cell centres x in [-52.5, 47.5] km, y in +-46.2 km) and every edge normal with
n_x >= 0. The cylinder sits at the origin as in his runs (176 cells inside), the wind is
+x, so mass_flx_me >= 0 on every edge, which is the condition his cell-local positive-
definite limiter needs (the generated grid has 440 edges with vn < 0, where his limiter
zeroes the flux). This module therefore adds the rows the generated grid cannot check: the
weight sets, the hybrid scheme and Jocksch's limiter; the 2/3/102 rows are regression rows
against the Fortran runs on this grid.

Reference data: weno_data/reference/jocksch_grid/<case>/error.txt with case names
ihadv<scheme>_hlim<limiter>[_dj1|_ones] (his printed '#' error, the neighbour-pair sum of
test_jocksch_cylinder.py); the results table is in icon-ajocksch/CAPTURE_NOTES.md.
"""

import math
import pathlib
import time as wall_time
from typing import Final

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection import tracer_advection, weno_least_squares
from icon4py.model.common import dimension as dims, model_backends, time
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import geometry_attributes as geom_attr, gridfile
from icon4py.model.common.initial_condition.analytical import moving_cylinder
from icon4py.model.driver import config as driver_config, driver, driver_utils

from .. import utils as test_utils
from ..fixtures import *  # noqa: F403
from .test_jocksch_cylinder import (
    _MASS_CONSERVATION_RTOL,
    CFL,
    CYLINDER_RADIUS,
    EXPECTED_GRID_SIZES,
    EXPERIMENT_CONFIG,
    N_TIME_STEPS,
    PAPER_TRUNCATION,
    WIND_ANGLE,
    WIND_SPEED,
    _neighbour_pair_error_sums,
)


#: Andreas Jocksch's own torus (a copy of his dispersion_relation/icon/grids/
#: torus_grid_r4_c200_elen100.nc; the name is misleading): 20 x 22, 5 km edges, centred
GRID_FILE: Final = pathlib.Path(
    "/capstor/scratch/cscs/cmueller/tracer_advection_port/icon-exclaim/weno_data/grids/"
    "jocksch_torus_grid_r4_c200_elen100.nc"
)
#: his cylinder is centred at the origin (the cell nearest to it is at (0, -1443.4) m)
CYLINDER_CENTER: Final[tuple[float, float]] = (0.0, 0.0)

_HADV = tracer_advection.HorizontalAdvectionType
_HLIM = tracer_advection.HorizontalAdvectionLimiter
_WEIGHTS = weno_least_squares.WenoLinearWeights
_Case = tuple[_HADV, _HLIM, _WEIGHTS]

#: Table 2 of the paper by (scheme, limiter, weight set): sqrt(pair sum / 3) printed with
#: three decimals (truncated in some entries, rounded in others, see CAPTURE_NOTES.md; the
#: gate is the union of both readings). The weight set only matters for 103 and 132.
#: "WENO d_j = 1" is the all-ones set (UNITY); the hybrid column is reproduced by the UNITY
#: set (3.3106), not by the optimised one (3.3124). The paper's 3.309 for "WENO d_j = 1 +
#: limiter" is not reproduced by the Fortran either (3.3084, PAPER_TABLE_2_NOT_REPRODUCED).
PAPER_TABLE_2: Final[dict[_Case, float]] = {
    (_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 4.023,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 3.402,
    (_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 3.859,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 3.058,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.UNITY): 3.310,
    (_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.UNITY): 3.310,
    (_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE, _WEIGHTS.OPTIMIZED): 3.842,
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.UNITY,
    ): 3.308,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE, _WEIGHTS.OPTIMIZED): 3.003,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC, _WEIGHTS.OPTIMIZED): 3.371,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE, _WEIGHTS.OPTIMIZED): 3.361,
    (_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC, _WEIGHTS.OPTIMIZED): 3.778,
}
#: paper entries the Fortran reference run itself does not hit (printed, not gated)
PAPER_TABLE_2_NOT_REPRODUCED: Final[dict[_Case, float]] = {
    # ihadv103_hlim4_ones prints 32.83643759358356, i.e. 3.3084; the paper says 3.309
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE, _WEIGHTS.UNITY): 3.309,
    # ihadv132_hlim0 (opt set) prints 32.91546685721701, i.e. 3.3124; the paper says 3.310
    (_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 3.310,
}

#: the '#' error the Fortran reference runs print on this grid,
#: weno_data/reference/jocksch_grid/<case>/error.txt (_ones = UNITY, _dj1 = HAND_TUNED)
FORTRAN_ERROR_SUM: Final[dict[_Case, float]] = {
    (_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 48.54601406791249,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 34.73851779887965,
    (_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 44.68350118207169,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 28.06422509932473,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.UNITY): 32.88393941867035,
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.HAND_TUNED): 32.24012112150874,
    (_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.UNITY): 32.88020252438776,
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID,
        _HLIM.NO_LIMITER,
        _WEIGHTS.OPTIMIZED,
    ): 32.91546685721701,
    (
        _HADV.LINEAR_2ND_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.OPTIMIZED,
    ): 44.27477978369598,
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.UNITY,
    ): 32.83643759358356,
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.UNITY,
    ): 32.83291364341792,
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.OPTIMIZED,
    ): 27.04602215516110,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC, _WEIGHTS.OPTIMIZED): 34.09527697903994,
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE, _WEIGHTS.OPTIMIZED): 33.90049401088558,
    (_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC, _WEIGHTS.OPTIMIZED): 42.82532613695046,
}
#: relative agreement with FORTRAN_ERROR_SUM, gated at about three times the value
#: measured on gtfn_cpu (in the comments) to leave room for the other backends. dt is
#: 1000 s here against 999.99999995 s there (the 1e-11 to 1e-10 of the non-WENO rows); the
#: WENO smoothness indicator and the hybrid's fit residual are single precision in the
#: Fortran and working precision here (the 1e-9 of the 103/132 rows, see
#: test_jocksch_reference.py). The test prints the difference per case.
FORTRAN_ERROR_RTOL: Final[dict[_Case, float]] = {
    (_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 1e-10,  # 3.0e-11
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 3e-10,  # 7.7e-11
    (_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 1e-9,  # 3.3e-10
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 2e-8,  # 4.0e-9
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.UNITY): 6e-9,  # 1.8e-9
    (_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.HAND_TUNED): 8e-9,  # 2.4e-9
    (_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.UNITY): 8e-9,  # 2.3e-9
    (_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED): 3e-9,  # 8.1e-10
    (
        _HADV.LINEAR_2ND_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.OPTIMIZED,
    ): 1e-9,  # 3.3e-10
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.UNITY,
    ): 6e-9,  # 2.0e-9
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.UNITY,
    ): 8e-9,  # 2.2e-9
    (
        _HADV.QUADRATIC_3RD_ORDER_WENO,
        _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
        _WEIGHTS.OPTIMIZED,
    ): 2e-8,  # 3.9e-9
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC, _WEIGHTS.OPTIMIZED): 1e-10,  # 3.2e-11
    (_HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE, _WEIGHTS.OPTIMIZED): 2e-10,  # 4.8e-11
    (_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC, _WEIGHTS.OPTIMIZED): 6e-10,  # 1.7e-10
}


def _run_one_period(
    *,
    horizontal_advection_type: _HADV,
    horizontal_advection_limiter: _HLIM,
    weno_linear_weights: _WEIGHTS,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> dict[str, float]:
    """One period of the cylinder on Jocksch's grid; the error measures of the final state."""
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
            "weno_linear_weights": weno_linear_weights,
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

    # the cylinder is a full disc of 176 cells at the origin, and his +x wind gives a
    # non-negative normal mass flux on every edge of this grid (n_x >= 0 everywhere)
    cylinder = moving_cylinder.sample_cylinder(
        config=ic_config,
        cell_center_x=cell_x,
        cell_center_y=cell_y,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    assert int(cylinder.sum()) == 176
    # mass_flx_me = u * n_x with u > 0, so n_x >= 0 on every edge is that condition
    edge_normal_x = geometry.get(geom_attr.EDGE_NORMAL_U).asnumpy()
    assert (edge_normal_x >= 0.0).all(), "his cell-local limiter needs mass_flx_me >= 0"

    qv_frames = test_utils.read_qv_frames(tmp_path)
    num_levels = experiment_config.vertical_grid.num_levels
    assert qv_frames.shape == (N_TIME_STEPS + 1, grid.num_cells, num_levels)
    assert np.isfinite(qv_frames).all()
    np.testing.assert_array_equal(qv_frames, np.broadcast_to(qv_frames[:, :, :1], qv_frames.shape))
    np.testing.assert_array_equal(qv_frames[0, :, 0], cylinder)

    # 100 steps of 1000 s at 1 m/s are exactly one period in x: the reference is the cylinder
    u, _ = moving_cylinder.compute_wind_components(ic_config)
    assert math.isclose(u * N_TIME_STEPS * dtime_seconds % domain_length, 0.0, abs_tol=1e-6)

    mass = np.einsum("tck,ck,c->t", qv_frames, airmass, cell_area)
    assert mass[0] > 0.0
    np.testing.assert_allclose(mass, mass[0], rtol=_MASS_CONSERVATION_RTOL)

    q_final = qv_frames[-1, :, 0]
    error = q_final - cylinder
    sum_squared_error = float(np.sum(error**2))
    jocksch_measure, all_pairs_measure = _neighbour_pair_error_sums(
        error=error,
        c2e2c=grid.get_connectivity(dims.C2E2C).asnumpy(),
        cell_x=cell_x,
        cell_y=cell_y,
        edge_length=edge_length,
    )
    np.testing.assert_allclose(all_pairs_measure, 3.0 * sum_squared_error, rtol=1e-12)
    return {
        "jocksch_measure": jocksch_measure,
        "all_pairs_measure": all_pairs_measure,
        "sum_squared_error": sum_squared_error,
        "overshoot_final": float(q_final.max() - 1.0),
        "undershoot_final": float(-q_final.min()),
        "overshoot_run": float(qv_frames.max() - 1.0),
        "undershoot_run": float(-qv_frames.min()),
        "relative_mass_change": float((mass[-1] - mass[0]) / mass[0]),
        "dtime_seconds": dtime_seconds,
        "num_levels": num_levels,
        "elapsed_wall_time": elapsed_wall_time,
    }


def _param(hadv: _HADV, hlim: _HLIM, weights: _WEIGHTS = _WEIGHTS.OPTIMIZED) -> object:
    weight_tag = (
        ""
        if hadv not in (_HADV.QUADRATIC_3RD_ORDER_WENO, _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID)
        else f"-{weights.name.lower()}"
    )
    return pytest.param(hadv, hlim, weights, id=f"ihadv{hadv.value}-hlim{hlim.value}{weight_tag}")


@pytest.mark.level("integration")
@pytest.mark.embedded_remap_error
@pytest.mark.skipif(not GRID_FILE.exists(), reason=f"Jocksch's grid file {GRID_FILE} not found")
@pytest.mark.parametrize(
    "horizontal_advection_type, horizontal_advection_limiter, weno_linear_weights",
    [
        # regression rows: the schemes the generated grid already gates
        _param(_HADV.LINEAR_2ND_ORDER, _HLIM.NO_LIMITER),
        _param(_HADV.QUADRATIC_3RD_ORDER, _HLIM.NO_LIMITER),
        _param(_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.NO_LIMITER),
        _param(_HADV.QUADRATIC_3RD_ORDER, _HLIM.MONOTONIC),
        _param(_HADV.QUADRATIC_3RD_ORDER, _HLIM.POSITIVE_DEFINITE),
        _param(_HADV.LINEAR_2ND_ORDER, _HLIM.MONOTONIC),
        # the weight sets of the quadratic WENO scheme
        _param(_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED),
        _param(_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.UNITY),
        _param(_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.NO_LIMITER, _WEIGHTS.HAND_TUNED),
        # the hybrid scheme
        _param(_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.UNITY),
        _param(_HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID, _HLIM.NO_LIMITER, _WEIGHTS.OPTIMIZED),
        # Jocksch's cell-local positive-definite limiter
        _param(_HADV.LINEAR_2ND_ORDER_WENO, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE),
        _param(_HADV.QUADRATIC_3RD_ORDER_WENO, _HLIM.CELL_LOCAL_POSITIVE_DEFINITE, _WEIGHTS.UNITY),
        _param(
            _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID,
            _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
            _WEIGHTS.UNITY,
        ),
        _param(
            _HADV.QUADRATIC_3RD_ORDER_WENO,
            _HLIM.CELL_LOCAL_POSITIVE_DEFINITE,
            _WEIGHTS.OPTIMIZED,
        ),
    ],
)
def test_jocksch_cylinder_one_period_on_jocksch_grid(
    horizontal_advection_type: _HADV,
    horizontal_advection_limiter: _HLIM,
    weno_linear_weights: _WEIGHTS,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    case: _Case = (horizontal_advection_type, horizontal_advection_limiter, weno_linear_weights)
    result = _run_one_period(
        horizontal_advection_type=horizontal_advection_type,
        horizontal_advection_limiter=horizontal_advection_limiter,
        weno_linear_weights=weno_linear_weights,
        tmp_path=tmp_path,
        process_props=process_props,
        backend=backend,
    )
    jocksch_measure = result["jocksch_measure"]
    fortran_value = FORTRAN_ERROR_SUM[case]
    relative_difference = abs(jocksch_measure - fortran_value) / fortran_value
    paper_value = PAPER_TABLE_2.get(case)
    paper_not_reproduced = PAPER_TABLE_2_NOT_REPRODUCED.get(case)
    print(
        f"\n{horizontal_advection_type.name} ({horizontal_advection_type.value}) + "
        f"{horizontal_advection_limiter.name} ({horizontal_advection_limiter.value}), "
        f"weights {weno_linear_weights.name}: {N_TIME_STEPS} steps of "
        f"dt = {result['dtime_seconds']} s, {result['num_levels']} level(s), "
        f"wall time {result['elapsed_wall_time']:.1f} s\n"
        f"  Jocksch measure (pairs within an edge length)  = {jocksch_measure:.6f}"
        f"  sqrt(/3) = {math.sqrt(jocksch_measure / 3.0):.6f}\n"
        f"  all neighbour pairs (= 3 sum e^2)              = {result['all_pairs_measure']:.6f}\n"
        f"  sum e^2                                        = {result['sum_squared_error']:.6f}\n"
        f"  overshoot (max q - 1) final / run              = {result['overshoot_final']:.6e} / "
        f"{result['overshoot_run']:.6e}\n"
        f"  undershoot (-min q) final / run                = {result['undershoot_final']:.6e} / "
        f"{result['undershoot_run']:.6e}\n"
        f"  relative mass change                           = {result['relative_mass_change']:.6e}\n"
        f"  Fortran reference (his grid)                   = {fortran_value}"
        f"  sqrt(/3) = {math.sqrt(fortran_value / 3.0):.6f}\n"
        f"  relative difference to the Fortran pair sum    = {relative_difference:.3e}\n"
        f"  paper Table 2                                  = {paper_value}"
        + (f" (paper prints {paper_not_reproduced}, not gated)" if paper_not_reproduced else "")
    )

    # the paper prints three decimals, truncated or rounded: gate on the union of both
    if paper_value is not None:
        root = math.sqrt(jocksch_measure / 3.0)
        assert paper_value - 0.5 * PAPER_TRUNCATION <= root < paper_value + PAPER_TRUNCATION, (
            f"sqrt(pair sum / 3) = {root:.6f} does not print as the paper's {paper_value}"
        )
    assert relative_difference <= FORTRAN_ERROR_RTOL[case]
