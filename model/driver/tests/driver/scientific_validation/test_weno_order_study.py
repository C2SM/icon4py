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
one test so that the distances can be formed; the slopes of every row are printed before
any band is checked. The measurements, the 8x member and the verdict on the claim are in
model/atmosphere/tracer_advection/docs/weno_idealized_status.md, section "W6"; the whole
study (six rows, factors 1, 2, 4) takes about 20 min on gtfn_cpu.

Differences from 'test_horizontal_advection_convergence': the family, the yaml (wider
Gaussian, 3 levels, quarter period, no limiter), the L2 norm, and the time step: here every
member runs at the yaml's CFL number ('_time_step'), so that the members are independent
and a finer one can be added on its own; the harness runs every member at the finest
member's step.

Environment overrides for the study's runs (unset in a normal test run):
'ICON4PY_WENO_ORDER_STUDY_FACTORS' (comma-separated refinement factors, e.g. "8" for one
finer member on a GPU), 'ICON4PY_WENO_ORDER_STUDY_ROWS' (comma-separated row ids),
'ICON4PY_WENO_ORDER_STUDY_RESULTS' (a JSON file every run's record is merged into as soon
as the run ends, one record per (row, factor): a run replaces only its own record and never
drops the others, so rows and factors can be run one pytest invocation at a time, e.g. one
batch job each; the final tracer of every run is saved next to the file) and
'ICON4PY_WENO_ORDER_STUDY_CHECK_ONLY' (set to 1: run nothing, evaluate the results file,
which is then required, for all rows or the selected ones). The distances to the pure row
are recomputed from the saved tracers whenever the results are evaluated, so the order in
which the rows ran does not matter. The slopes are fitted over the factors
'_REFINEMENT_FACTORS' and, if the file holds more (an 8x member), also over all of them, and
printed for every row. The gates (see '_ROWS') are checked on every evaluated row whose
records hold '_REFINEMENT_FACTORS' (in check-only mode every evaluated row must hold them):
the '_REFINEMENT_FACTORS' fit for 2, 3 and 102, the local rate between the last two members
in the file for 3 and for the quadratic WENO rows, whose gates document a known deficiency
of the published type-VI construction rather than an order of accuracy.
"""

import dataclasses
import json
import math
import os
import pathlib
import socket
import time as wall_time
from collections.abc import Callable, Mapping
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

_ERROR_KEYS: Final = ("error_l1", "error_l2", "error_linf")
_DISTANCE_KEYS: Final = ("distance_to_miura3_l2", "distance_to_miura3_linf")


#: acceptable ranges of the (L1, L2, Linf) rates
type _Bands = tuple[list[float], list[float], list[float]]


@dataclasses.dataclass(frozen=True)
class _Row:
    id: str
    advection_type: tracer_advection.HorizontalAdvectionType
    linear_weights: weno_least_squares.WenoLinearWeights
    #: acceptable slopes of the fit over _REFINEMENT_FACTORS; None: the fit is not gated
    fit_bands: _Bands | None
    #: acceptable local rates between the last two members in the results file, keyed by
    #: their factors; empty: the local rate is not gated
    last_rate_bands: Mapping[tuple[int, int], _Bands] = dataclasses.field(default_factory=dict)


#: the formal order of the pure quadratic scheme, which it meets on this family
_THIRD_ORDER_BAND: Final = [
    harness._THIRD_ORDER - harness._TOL,
    harness._THIRD_ORDER + harness._TOL,
]

#: the member pairs the study has run: consecutive factors of 1, 2, 4, 8, 16
_MEMBER_PAIRS: Final = ((2, 4), (4, 8), (8, 16))


def _measured_local_rates(l1: float, l2: float, linf: float) -> _Bands:
    """Bands at the formal-order tolerance around local rates measured between two members."""

    def band(rate: float) -> list[float]:
        return [rate - harness._TOL, rate + harness._TOL]

    return band(l1), band(l2), band(linf)


#: Gates, measured in W6s (2026-09-11): 1x-4x on gtfn_cpu, 8x and 16x on dace_gpu. Backend
#: agreement was checked on the common members 1x-4x only (relative errors within 1.72e-12,
#: final tracers within 1.55e-15); the 8x and 16x members exist on dace_gpu only.
#:
#: - miura3 is gated on its formal order, 3 +- 0.1, on the (1, 2, 4) fit and on the local rate
#:   between the last two members.
#: - miura and miura_weno are regression guards on the (1, 2, 4) fit, centred on the gtfn_cpu
#:   measurement at the harness's measured-rate width (harness._measured), not order
#:   statements.
#: - The quadratic WENO rows (103 OPTIMIZED, 103 UNITY, 132) document a known deficiency of
#:   the published type-VI construction, not a pre-asymptotic range: the type-VI candidates
#:   A+_full - sum d_i A+_i (mo_intp_coeffs_lsq_bln.f90 2669-2680 on
#:   transport_ajocksch_capture, ported literally in
#:   weno_least_squares.compute_weno_pseudoinverse_quadratic) return (1 - S) times the
#:   derivatives of smooth data, so the blend returns them short by a constant delta
#:   (-1.6214e-3 OPTIMIZED, -4.1647e-4 UNITY, -1.2127e-3 hybrid), a first-order diffusion
#:   that takes over as the grid is refined. They are gated on the local rate between the
#:   last two members in the results file, at +- 0.1 around the rate measured for that pair,
#:   every band below the third-order one, so the gate tells first from third order and
#:   fails if the construction changes; their (1, 2, 4) fit, whose curvature is that
#:   transition, is printed only. The sharp check of the mechanism is the numpy unit test
#:   model/atmosphere/tracer_advection/tests/tracer_advection/unit_tests/
#:   test_weno_type_vi_bias.py; the finding: docs/weno_idealized_status.md, "W6".
_ROWS: Final[tuple[_Row, ...]] = (
    # measured L1 2.182 +- 0.048, L2 2.225 +- 0.058, Linf 2.319 +- 0.085
    _Row(
        "miura",
        _MIURA,
        _OPTIMIZED,
        (harness._measured(2.18), harness._measured(2.23), harness._measured(2.32)),
    ),
    # measured L1 2.977 +- 0.009, L2 2.967 +- 0.013, Linf 2.960 +- 0.016;
    # local rates (L1, L2, Linf) x2-x4 2.99, 2.99, 2.99; x4-x8 3.00, 3.00, 3.00;
    # x8-x16 3.00, 3.00, 3.00 (x1,2,4,8,16 fit: 2.989, 2.985, 2.984)
    _Row(
        "miura3",
        _MIURA3,
        _OPTIMIZED,
        (_THIRD_ORDER_BAND, _THIRD_ORDER_BAND, _THIRD_ORDER_BAND),
        {pair: (_THIRD_ORDER_BAND, _THIRD_ORDER_BAND, _THIRD_ORDER_BAND) for pair in _MEMBER_PAIRS},
    ),
    # measured L1 2.409 +- 0.030, L2 1.966 +- 0.046, Linf 1.298 +- 0.034
    _Row(
        "miura_weno",
        _MIURA_WENO,
        _OPTIMIZED,
        (harness._measured(2.41), harness._measured(1.97), harness._measured(1.30)),
    ),
    # first order below 200 m edges; fit x1,2,4: L1 1.610 +- 0.271, L2 1.665 +- 0.267, Linf
    # 1.754 +- 0.288; x1,2,4,8,16: 1.267, 1.299, 1.344; local rates (L1, L2, Linf) x1-x2 2.08,
    # 2.13, 2.25, and the gated ones below
    _Row(
        "miura3_weno_opt",
        _MIURA3_WENO,
        _OPTIMIZED,
        None,
        {
            (2, 4): _measured_local_rates(1.14, 1.20, 1.26),
            (4, 8): _measured_local_rates(1.03, 1.04, 1.05),
            (8, 16): _measured_local_rates(1.01, 1.01, 1.01),
        },
    ),
    # losing order later than OPTIMIZED (its delta is 3.9x smaller); fit x1,2,4: L1 2.827 +-
    # 0.039, L2 2.815 +- 0.057, Linf 2.820 +- 0.053; x1,2,4,8,16: 2.354, 2.357, 2.393; local
    # rates x1-x2 2.90, 2.91, 2.91, and the gated ones below
    _Row(
        "miura3_weno_unity",
        _MIURA3_WENO,
        _UNITY,
        None,
        {
            (2, 4): _measured_local_rates(2.76, 2.72, 2.73),
            (4, 8): _measured_local_rates(2.20, 2.18, 2.24),
            (8, 16): _measured_local_rates(1.43, 1.52, 1.60),
        },
    ),
    # losing order (delta -1.2127e-3 in its WENO branch); fit x1,2,4: L1 2.577 +- 0.120, L2
    # 2.556 +- 0.128, Linf 2.580 +- 0.116; x1,2,4,8: 2.254, 2.262, 2.260; local rates x1-x2
    # 2.78, 2.78, 2.78, and the gated ones below (no 16x member); the WENO branch blends with
    # unit weights at run time (f90 3684) on candidates assembled with this row's (optimised)
    # set
    _Row(
        "miura3_weno_hybrid",
        _MIURA3_WENO_HYBRID,
        _OPTIMIZED,
        None,
        {
            (2, 4): _measured_local_rates(2.37, 2.34, 2.38),
            (4, 8): _measured_local_rates(1.57, 1.65, 1.58),
        },
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


def _load_records(results_path: pathlib.Path) -> dict[str, dict[str, dict]]:
    if not results_path.exists():
        return {}
    return json.loads(results_path.read_text()).get("records", {})


def _write_results(results_path: pathlib.Path, payload: dict) -> None:
    """Replace the results file atomically, so that a killed job cannot leave half a file."""
    partial = results_path.with_name(results_path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=1))
    partial.replace(results_path)


def _merge_record(results_path: pathlib.Path, row_id: str, factor: int, record: dict) -> None:
    """Set one (row, factor) record in the results file, keeping every other record.

    The file is read again right before the write rather than taken from the start of the
    invocation, so records written in between (another invocation) are kept as well.
    """
    records = _load_records(results_path)
    records.setdefault(row_id, {})[str(factor)] = record
    _write_results(results_path, {"records": records})


def _distances(simulated: np.ndarray, pure_tracer: np.ndarray) -> dict[str, float]:
    return {
        "distance_to_miura3_l2": _relative_l2_error(simulated, pure_tracer),
        "distance_to_miura3_linf": float(
            np.max(np.abs(simulated - pure_tracer)) / np.max(np.abs(pure_tracer))
        ),
    }


def _saved_tracer(results_path: pathlib.Path, row_id: str, factor: int) -> np.ndarray | None:
    path = _tracer_path(results_path, row_id, factor)
    return np.load(path) if path.exists() else None


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
        "backend": getattr(backend, "name", str(backend)),
        "host": socket.gethostname(),
    }
    if row.advection_type is _MIURA3_WENO_HYBRID:
        # the selection mask of the last step (computed from the tracer at its start): the
        # (cell, level) points whose edges took the WENO flux
        horizontal = icon4py_driver.granules.tracer_advection._horizontal_advection
        use_weno = data_alloc.as_numpy(horizontal._tracer_flux._use_weno.ndarray)
        record["weno_cell_fraction"] = float(np.mean(use_weno))
    if pure_tracer is not None:
        record.update(_distances(simulated, pure_tracer))
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


def _local_rates(grid_spacing: list[float], values: list[float]) -> list[float]:
    """The rate between neighbouring members; a straight line has them all equal."""
    return [
        math.log(values[i] / values[i + 1]) / math.log(grid_spacing[i] / grid_spacing[i + 1])
        if values[i] > 0.0 and values[i + 1] > 0.0
        else float("nan")
        for i in range(len(values) - 1)
    ]


def _refresh_distances(results_path: pathlib.Path, records: dict[str, dict[str, dict]]) -> None:
    """Recompute every WENO record's distance to the pure row from the saved tracers."""
    for row_id, per_factor in records.items():
        if row_id == _PURE_ROW:
            continue
        for factor, record in per_factor.items():
            simulated = _saved_tracer(results_path, row_id, int(factor))
            pure_tracer = _saved_tracer(results_path, _PURE_ROW, int(factor))
            if simulated is not None and pure_tracer is not None:
                record.update(_distances(simulated, pure_tracer))


def _fits(per_factor: dict[str, dict], factors: tuple[int, ...]) -> dict[str, dict]:
    """Slope +- stderr and local rates of every norm (and distance) over the given factors."""
    records = [per_factor[str(factor)] for factor in factors]
    grid_spacing = [record["mean_edge_length"] for record in records]
    keys = list(_ERROR_KEYS)
    if all(key in record for record in records for key in _DISTANCE_KEYS):
        keys += _DISTANCE_KEYS
    return {
        key: {
            "slope": _fit_slope(grid_spacing, [record[key] for record in records]),
            "local_rates": _local_rates(grid_spacing, [record[key] for record in records]),
        }
        for key in keys
    }


def _check_last_local_rates(row: _Row, per_factor: dict[str, dict]) -> None:
    """Gate the (L1, L2, Linf) local rates between the last two members in the records."""
    factor_pair = tuple(sorted(int(factor) for factor in per_factor)[-2:])
    bands = row.last_rate_bands.get(factor_pair)
    assert bands is not None, (
        f"{row.id}: no local-rate bands for the members {factor_pair}; "
        f"recorded for {sorted(row.last_rate_bands)}"
    )
    if row.id != _PURE_ROW:
        # the point of these gates: every band stays clear of the formal third order
        assert all(band[1] < _THIRD_ORDER_BAND[0] for band in bands), (
            f"{row.id}: a band of {factor_pair} reaches the third-order band"
        )
    records = [per_factor[str(factor)] for factor in factor_pair]
    grid_spacing = [record["mean_edge_length"] for record in records]
    for key, band in zip(_ERROR_KEYS, bands, strict=True):
        (rate,) = _local_rates(grid_spacing, [record[key] for record in records])
        print(f"{row.id} x{factor_pair[0]}-x{factor_pair[1]} local {key} rate {rate:.4f}")
        assert band[0] <= rate <= band[1], (
            f"{row.id}: local {key} rate {rate:.4f} between x{factor_pair[0]} and "
            f"x{factor_pair[1]} outside {band}"
        )


def _evaluate(
    results_path: pathlib.Path, rows: tuple[_Row, ...], *, require_all: bool, write: bool
) -> None:
    """Print the slopes of every row in the results file, then check the gates of 'rows'.

    A row is gated when its records hold '_REFINEMENT_FACTORS': on that fit if it has
    'fit_bands', on the local rate between its last two members if it has 'last_rate_bands'
    (a pair without bands fails). With 'require_all' a row of 'rows' without those factors
    fails the test instead of being skipped.
    """
    records = _load_records(results_path)
    _refresh_distances(results_path, records)
    refinement_label = ",".join(str(factor) for factor in _REFINEMENT_FACTORS)
    slopes: dict[str, dict[str, dict]] = {}
    for row in _ROWS:
        per_factor = records.get(row.id, {})
        present = tuple(sorted(int(factor) for factor in per_factor))
        if not set(_REFINEMENT_FACTORS) <= set(present):
            if per_factor:
                print(f"\nno slopes, {row.id}: factors {present} only", flush=True)
            continue
        fit_sets = [_REFINEMENT_FACTORS] + ([present] if present != _REFINEMENT_FACTORS else [])
        slopes[row.id] = {}
        for factors in fit_sets:
            label = ",".join(str(factor) for factor in factors)
            fits = _fits(per_factor, factors)
            slopes[row.id][label] = fits
            print(
                f"\nslopes {row.id:18s} x{label}: "
                + ", ".join(
                    f"{key.removeprefix('error_').replace('distance_to_miura3_', '|q-q3| ')} "
                    f"{fit['slope'][0]:.3f} +- {fit['slope'][1]:.3f} "
                    f"(local {', '.join(f'{rate:.2f}' for rate in fit['local_rates'])})"
                    for key, fit in fits.items()
                ),
                flush=True,
            )
    if write:
        _write_results(results_path, {"records": records, "slopes": slopes})

    for row in rows:
        if row.id not in slopes:
            message = f"{row.id}: no records for all of {_REFINEMENT_FACTORS} in {results_path}"
            assert not require_all, message
            print(f"\nnot gated, {message}", flush=True)
            continue
        if row.fit_bands is not None:
            l1_band, l2_band, linf_band = row.fit_bands
            per_factor = [records[row.id][str(factor)] for factor in _REFINEMENT_FACTORS]
            harness._check_convergence(
                l1_acceptable_range=l1_band,
                linf_acceptable_range=linf_band,
                error_l1=[record["error_l1"] for record in per_factor],
                error_linf=[record["error_linf"] for record in per_factor],
                grid_spacing=[record["mean_edge_length"] for record in per_factor],
            )
            slope_l2, stderr_l2 = slopes[row.id][refinement_label]["error_l2"]["slope"]
            assert l2_band[0] <= slope_l2 <= l2_band[1], (
                f"{row.id}: L2 rate {slope_l2:.4f} outside {l2_band}"
            )
            assert stderr_l2 <= harness._STD_TOL
        if row.last_rate_bands:
            _check_last_local_rates(row, records[row.id])


@pytest.mark.level("validation")
@pytest.mark.embedded_remap_error
def test_weno_order_study(
    *,
    tmp_path: pathlib.Path,
    generate_torus_grid: Callable[..., pathlib.Path],
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    factors = _selected_factors()
    rows = _selected_rows()
    if os.environ.get("ICON4PY_WENO_ORDER_STUDY_CHECK_ONLY", "0") not in ("", "0"):
        results = os.environ.get("ICON4PY_WENO_ORDER_STUDY_RESULTS")
        assert results, "the check-only mode evaluates ICON4PY_WENO_ORDER_STUDY_RESULTS"
        assert pathlib.Path(results).exists(), f"no results file {results}"
        _evaluate(pathlib.Path(results), rows, require_all=True, write=False)
        return

    allocator = model_backends.get_allocator(backend)
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

    # rows in _ROWS order, so the pure row's tracers are saved before a WENO row reads them
    for row in rows:
        for factor in factors:
            record, simulated = _run_member(
                row=row,
                factor=factor,
                base_config=base_config,
                grid_manager=grid_managers[factor],
                process_props=process_props,
                backend=backend,
                pure_tracer=(
                    _saved_tracer(results_path, _PURE_ROW, factor) if row.id != _PURE_ROW else None
                ),
            )
            np.save(_tracer_path(results_path, row.id, factor), simulated)
            _merge_record(results_path, row.id, factor, record)
            print("\n" + _describe(row, factor, record), flush=True)

    _evaluate(results_path, rows, require_all=False, write=True)
