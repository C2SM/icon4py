# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Convergence order of the quadratic WENO scheme (103) and its hybrid (132) on a smooth profile.

Jocksch et al. (PPAM 2026) give no convergence study; they claim that "for smooth solutions
the WENO discretisation delivers identical results to the pure scheme" (sections 2.3, 4).
This study measures that claim on the Gaussian of
'experiment_configs/weno_order_study_gaussian_2d.yaml' with the convergence harness of
'linear_advection_tests.py', on a family of torus grids whose finest member is one
bisection coarser than that harness's '_COARSE_TORUS_FAMILY' (the 27-candidate loop of
103 is what made the harness skip its 103 rows, see '_MIURA3_WENO_SKIP' there). Per row
(scheme) and resolution it records the relative L1, L2 and L-infinity errors against the
analytic reference, the slope of every norm with its standard error, the direct distance
between the WENO schemes and the pure quadratic scheme (3) in L2 and L-infinity with its
slope, and for the hybrid the fraction of cells that took the WENO branch. All rows share
one test so that the distances can be formed; the bands are checked after every row has
been measured and printed.

Differences from 'test_horizontal_advection_convergence': the family, the yaml (wider
Gaussian, 3 levels, quarter period, no limiter), the L2 norm, and the time step: here every
member runs at the yaml's CFL number ('_time_step'), so that the members are independent
and a finer one can be added on its own; the harness runs every member at the finest
member's step.

Environment overrides for the study's runs (unset in a normal test run):
'ICON4PY_WENO_ORDER_STUDY_FACTORS' (comma-separated refinement factors, e.g. "8" for one
finer member on a GPU), 'ICON4PY_WENO_ORDER_STUDY_ROWS' (comma-separated row ids) and
'ICON4PY_WENO_ORDER_STUDY_RESULTS' (a JSON file the results are merged into after every
run; the final tracer of every run is saved next to it, and a run without the pure row takes
that row's saved tracers for the distances, so the rows can be run one pytest invocation
at a time). With a subset of factors or rows the bands are not checked.
"""

import dataclasses
import json
import os
import pathlib
import time as wall_time
from collections.abc import Callable
from typing import Final

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest
from scipy.stats import linregress

from icon4py.model.atmosphere.tracer_advection import tracer_advection, weno_least_squares
from icon4py.model.common import model_backends, time
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import geometry_attributes as geometry_meta
from icon4py.model.common.initial_condition.analytical import linear_horizontal_advection
from icon4py.model.common.states import factory as states_factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import config as test_config

from ..fixtures import *  # noqa: F403
from . import linear_advection_tests as harness


_EXPERIMENT_CASE: Final = "weno_order_study_gaussian_2d"

#: (n_rows, n_cols, edge_length) of the coarsest member: the harness's _COARSE_TORUS_FAMILY
#: one bisection coarser (3344 / 13376 / 53504 cells), the same domain 17.6 km x 13.2 km.
#: The Gaussian's e-folding radius (decay_radius 0.35 of the domain height, 1e-3 at that
#: radius, so r_e = 0.35 H / sqrt(ln 1000) = 1753 m) is 4.4 edge lengths / 5.1 rows of
#: cells / 6.7 cell diameters (sqrt of the cell area) on the coarsest member.
_TORUS_FAMILY: Final[harness.TorusFamily] = (38, 44, 400.0)
_REFINEMENT_FACTORS: Final = (1, 2, 4)

_MIURA = tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER
_MIURA3 = tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER
_MIURA_WENO = tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER_WENO
_MIURA3_WENO = tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER_WENO
_MIURA3_WENO_HYBRID = tracer_advection.HorizontalAdvectionType.QUADRATIC_3RD_ORDER_WENO_HYBRID
_OPTIMIZED = weno_least_squares.WenoLinearWeights.OPTIMIZED
_UNITY = weno_least_squares.WenoLinearWeights.UNITY

#: the row the WENO rows are measured against
_PURE_ROW: Final = "miura3"


@dataclasses.dataclass(frozen=True)
class _Row:
    id: str
    advection_type: tracer_advection.HorizontalAdvectionType
    linear_weights: weno_least_squares.WenoLinearWeights
    #: acceptable slope bands (L1, L2, Linf); _MEASURE_ONLY until measured
    l1_band: list[float]
    l2_band: list[float]
    linf_band: list[float]


_ROWS: Final[tuple[_Row, ...]] = (
    _Row(
        "miura",
        _MIURA,
        _OPTIMIZED,
        harness._MEASURE_ONLY,
        harness._MEASURE_ONLY,
        harness._MEASURE_ONLY,
    ),
    _Row(
        "miura3",
        _MIURA3,
        _OPTIMIZED,
        harness._MEASURE_ONLY,
        harness._MEASURE_ONLY,
        harness._MEASURE_ONLY,
    ),
    _Row(
        "miura_weno",
        _MIURA_WENO,
        _OPTIMIZED,
        harness._MEASURE_ONLY,
        harness._MEASURE_ONLY,
        harness._MEASURE_ONLY,
    ),
    _Row(
        "miura3_weno_opt",
        _MIURA3_WENO,
        _OPTIMIZED,
        harness._MEASURE_ONLY,
        harness._MEASURE_ONLY,
        harness._MEASURE_ONLY,
    ),
    _Row(
        "miura3_weno_unity",
        _MIURA3_WENO,
        _UNITY,
        harness._MEASURE_ONLY,
        harness._MEASURE_ONLY,
        harness._MEASURE_ONLY,
    ),
    _Row(
        "miura3_weno_hybrid",
        _MIURA3_WENO_HYBRID,
        _OPTIMIZED,
        harness._MEASURE_ONLY,
        harness._MEASURE_ONLY,
        harness._MEASURE_ONLY,
    ),
)


def _relative_l2_error(simulated: np.ndarray, reference: np.ndarray) -> float:
    """The L2 companion of the harness's '_compute_relative_errors' (L1 and Linf)."""
    return float(np.sqrt(np.sum((simulated - reference) ** 2) / np.sum(reference**2)))


def _fit_slope(grid_spacing: list[float], errors: list[float]) -> tuple[float, float]:
    """Slope of log(error) against log(h) and its standard error; NaN with < 3 points."""
    if len(grid_spacing) < 3 or not all(e > 0.0 for e in errors):
        return float("nan"), float("nan")
    fit = linregress(np.log(grid_spacing), np.log(errors))
    return float(fit.slope), float(fit.stderr)


def _tracer_path(results_path: pathlib.Path, row_id: str, factor: int) -> pathlib.Path:
    return results_path.with_name(f"{results_path.stem}_{row_id}_x{factor}.npy")


def _selected_factors() -> tuple[int, ...]:
    override = os.environ.get("ICON4PY_WENO_ORDER_STUDY_FACTORS")
    if not override:
        return _REFINEMENT_FACTORS
    return tuple(int(f) for f in override.split(","))


def _selected_rows() -> tuple[_Row, ...]:
    override = os.environ.get("ICON4PY_WENO_ORDER_STUDY_ROWS")
    if not override:
        return _ROWS
    wanted = override.split(",")
    unknown = set(wanted) - {row.id for row in _ROWS}
    if unknown:
        raise ValueError(f"unknown rows {sorted(unknown)}; known: {[row.id for row in _ROWS]}")
    return tuple(row for row in _ROWS if row.id in wanted)


def _time_step(
    config: driver_config.ExperimentConfig, grid_manager, edge_length: float
) -> tuple[float, int]:
    """The member's own time step at the yaml's CFL number and the step count of the run.

    The harness derives one step from the finest member and runs every member with it;
    here the members are independent (constant CFL), so a finer one can be added alone.
    """
    initial_condition = config.initial_condition.config
    assert type(initial_condition) is linear_horizontal_advection.LinearHorizontalAdvectionConfig
    domain_length = grid_manager.grid.grid_params.domain_length
    domain_height = grid_manager.grid.grid_params.domain_height
    assert domain_length is not None and domain_height is not None
    vel_max = linear_horizontal_advection.compute_max_velocity(
        velocity_field=initial_condition.velocity_field,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    match config.driver.end_of_simulation:
        case time.RelativeTime() as relative:
            integration_time = relative.total_seconds()
        case _:
            raise ValueError("end_of_simulation must be a RelativeTime for this study")
    # the nominal edge length rather than the measured mean, so that the step does not
    # depend on which members are in the run (the two agree to round-off on these grids)
    dtime = min(initial_condition.cfl_number * edge_length / vel_max, integration_time)
    return dtime, int(integration_time / dtime)


def _previous_results(
    results_path: pathlib.Path, rows: tuple[_Row, ...], factors: tuple[int, ...]
) -> tuple[dict[str, dict[str, dict]], dict[str, dict[int, np.ndarray]]]:
    """The records of earlier invocations and the pure row's saved tracers, if any."""
    records: dict[str, dict[str, dict]] = {}
    if results_path.exists():
        records = json.loads(results_path.read_text()).get("records", {})
    for row in rows:
        records[row.id] = {}
    tracers: dict[str, dict[int, np.ndarray]] = {row.id: {} for row in rows}
    if _PURE_ROW not in tracers:
        tracers[_PURE_ROW] = {
            factor: np.load(_tracer_path(results_path, _PURE_ROW, factor))
            for factor in factors
            if _tracer_path(results_path, _PURE_ROW, factor).exists()
        }
    return records, tracers


def _run_member(
    *,
    row: _Row,
    factor: int,
    base_config: driver_config.ExperimentConfig,
    grid_manager,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    pure_tracer: np.ndarray | None,
) -> tuple[dict, np.ndarray]:
    """One driver run; its record (errors, wall time, distances, mask) and final tracer."""
    edge_length = _TORUS_FAMILY[2] / factor
    dtime, num_steps = _time_step(base_config, grid_manager, edge_length)
    config = base_config.with_overrides(
        driver={
            "dtime": time.RelativeTime(seconds=dtime),
            "end_of_simulation": time.NumTimeSteps(num_steps),
        },
        tracer_advection={
            "horizontal_advection_type": row.advection_type,
            "weno_linear_weights": row.linear_weights,
        },
    )
    # RelativeTime is a datetime.timedelta, which rounds the step to microseconds
    # (500.4966 us -> 500 us on the 4x member); the reference has to be evaluated at
    # the time the driver integrated to, as the harness does, not at num_steps * dtime
    # (measured: a 0.1 % step mismatch puts the 4x error of scheme 2 above its 2x error)
    dtime = config.driver.dtime.total_seconds()
    start = wall_time.perf_counter()
    ds, icon4py_driver = driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )
    elapsed = wall_time.perf_counter() - start

    simulated = data_alloc.as_numpy(ds.tracers.current.qv.ndarray)
    initial_condition = config.initial_condition.config
    assert type(initial_condition) is linear_horizontal_advection.LinearHorizontalAdvectionConfig
    reference = data_alloc.as_numpy(
        linear_horizontal_advection.construct_reference_tracer(
            config=initial_condition,
            grid=grid_manager.grid,
            static_fields=icon4py_driver.static_field_factories,
            integration_time=num_steps * dtime,
            num_levels=config.vertical_grid.num_levels,
        )
    )
    error_l1, error_linf = harness._compute_relative_errors(simulated, reference)
    record = {
        "cells": int(simulated.shape[0]),
        "num_steps": num_steps,
        "dtime": dtime,
        "mean_edge_length": float(
            icon4py_driver.static_field_factories.geometry.get(
                geometry_meta.MEAN_EDGE_LENGTH, states_factory.RetrievalType.SCALAR
            )
        ),
        "error_l1": float(error_l1),
        "error_l2": _relative_l2_error(simulated, reference),
        "error_linf": float(error_linf),
        "wall_time_s": elapsed,
    }
    if row.advection_type is _MIURA3_WENO_HYBRID:
        # the mask of the last step, the cells whose edges took the WENO flux
        horizontal = icon4py_driver.granules.tracer_advection._horizontal_advection
        use_weno = data_alloc.as_numpy(horizontal._use_weno.ndarray)
        record["weno_cell_fraction"] = float(np.mean(use_weno))
    if pure_tracer is not None:
        record["distance_to_miura3_l2"] = _relative_l2_error(simulated, pure_tracer)
        record["distance_to_miura3_linf"] = float(
            np.max(np.abs(simulated - pure_tracer)) / np.max(np.abs(pure_tracer))
        )
    return record, simulated


def _describe(row: _Row, factor: int, record: dict) -> str:
    line = (
        f"{row.id:18s} x{factor}: {record['cells']:6d} cells, {record['num_steps']} steps, "
        f"L1 {record['error_l1']:.4e} L2 {record['error_l2']:.4e} "
        f"Linf {record['error_linf']:.4e}, {record['wall_time_s']:.1f} s"
    )
    if "weno_cell_fraction" in record:
        line += f", WENO cells {record['weno_cell_fraction']:.4f}"
    if "distance_to_miura3_l2" in record:
        line += (
            f", |q - q_3| L2 {record['distance_to_miura3_l2']:.3e} "
            f"Linf {record['distance_to_miura3_linf']:.3e}"
        )
    return line


def _row_slopes(
    per_factor: list[dict], grid_spacing: list[float]
) -> dict[str, tuple[float, float]]:
    keys = ["error_l1", "error_l2", "error_linf"]
    if all("distance_to_miura3_l2" in record for record in per_factor):
        keys += ["distance_to_miura3_l2", "distance_to_miura3_linf"]
    return {key: _fit_slope(grid_spacing, [record[key] for record in per_factor]) for key in keys}


@pytest.mark.level("validation")
@pytest.mark.embedded_remap_error
def test_weno_order_study(
    *,
    tmp_path: pathlib.Path,
    generate_torus_grid: Callable[..., pathlib.Path],
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)
    factors = _selected_factors()
    rows = _selected_rows()
    full_study = factors == _REFINEMENT_FACTORS and rows == _ROWS
    results_path = pathlib.Path(
        os.environ.get("ICON4PY_WENO_ORDER_STUDY_RESULTS", tmp_path / "weno_order_study.json")
    )
    base_config = config_io.read_yaml_str(
        (test_config.EXPERIMENT_CONFIG_PATH / f"{_EXPERIMENT_CASE}.yaml").read_text(),
        driver_config.ExperimentConfig,
    ).with_overrides(driver={"output_path": tmp_path / "ci_driver_output"})

    base_rows, base_cols, base_edge_length = _TORUS_FAMILY
    grid_managers = {
        factor: driver_utils.create_grid_manager(
            grid_file_path=generate_torus_grid(
                n_rows=base_rows * factor,
                n_cols=base_cols * factor,
                edge_length=base_edge_length / factor,
            ),
            vertical_grid_config=base_config.vertical_grid,
            allocator=allocator,
            process_props=process_props,
        )
        for factor in factors
    }

    records, tracers = _previous_results(results_path, rows, factors)
    for row in rows:
        for factor in factors:
            record, simulated = _run_member(
                row=row,
                factor=factor,
                base_config=base_config,
                grid_manager=grid_managers[factor],
                process_props=process_props,
                backend=backend,
                pure_tracer=tracers[_PURE_ROW].get(factor) if row.id != _PURE_ROW else None,
            )
            records[row.id][str(factor)] = record
            tracers[row.id][factor] = simulated
            np.save(_tracer_path(results_path, row.id, factor), simulated)
            print("\n" + _describe(row, factor, record), flush=True)
            results_path.write_text(json.dumps({"records": records}, indent=1))

    # the slopes, printed for every row before anything is asserted
    grid_spacing = [records[rows[0].id][str(factor)]["mean_edge_length"] for factor in factors]
    slopes = {
        row.id: _row_slopes([records[row.id][str(factor)] for factor in factors], grid_spacing)
        for row in rows
    }
    for row in rows:
        print(
            f"\nslopes {row.id:18s}: "
            + ", ".join(
                f"{key.removeprefix('error_').removeprefix('distance_to_miura3_')} "
                f"{slope:.3f} +- {stderr:.3f}"
                for key, (slope, stderr) in slopes[row.id].items()
            ),
            flush=True,
        )
    results_path.write_text(json.dumps({"records": records, "slopes": slopes}, indent=1))

    if not full_study:
        return
    for row in rows:
        per_factor = [records[row.id][str(factor)] for factor in factors]
        harness._check_convergence(
            l1_acceptable_range=row.l1_band,
            linf_acceptable_range=row.linf_band,
            error_l1=[record["error_l1"] for record in per_factor],
            error_linf=[record["error_linf"] for record in per_factor],
            grid_spacing=grid_spacing,
        )
        slope_l2, stderr_l2 = slopes[row.id]["error_l2"]
        assert row.l2_band[0] <= slope_l2 <= row.l2_band[1], (
            f"{row.id}: L2 rate {slope_l2:.4f} outside {row.l2_band}"
        )
        assert stderr_l2 <= harness._STD_TOL
