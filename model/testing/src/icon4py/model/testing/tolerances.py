# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Machinery for measuring and tightening numeric test tolerances.

'assert_dallclose' (in 'test_utils') can, instead of asserting, record the measured maximum
absolute and relative differences between actual and reference fields. The 'ToleranceRecorder'
collects these measurements together with the current test context (test id, backend, experiment)
so that a downstream script can propose tighter tolerances. It can also collect non-failing
warnings about tolerances that have become much larger than the measured difference.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import json
import math
import pathlib

import gt4py.next as gtx
import numpy as np
import numpy.typing as npt

from icon4py.model.common import dimension as dims


__all__ = [
    "DETERMINISTIC_CPU_BACKENDS",
    "DRIFT_FACTOR",
    "SAFETY_FACTOR",
    "DriftWarning",
    "Measurement",
    "ToleranceRecorder",
    "activate_recorder",
    "aggregate_max_abs",
    "deactivate_recorder",
    "get_active_recorder",
    "load_dimension_keyed_tolerances",
    "load_measurements",
    "max_differences",
    "propose_tolerance",
]

# A stored tolerance is flagged as drifted (too loose) when it exceeds the measured difference by
# more than this factor.
DRIFT_FACTOR = 100.0
# Proposed tolerances are the measured difference scaled by this factor to leave headroom.
SAFETY_FACTOR = 4.0
# Backends whose results are deterministic enough that measured differences may be used to tighten
# tolerances automatically. GPU and dace backends are excluded (documented as non-deterministic).
DETERMINISTIC_CPU_BACKENDS = frozenset({"gtfn_cpu", "embedded"})


def max_differences(actual: npt.ArrayLike, desired: npt.ArrayLike) -> tuple[float, float]:
    """Return the maximum absolute and relative difference between 'actual' and 'desired'."""
    actual_array = np.asarray(actual, dtype=float)
    desired_array = np.asarray(desired, dtype=float)
    absolute = np.abs(actual_array - desired_array)
    max_absolute = float(np.max(absolute)) if absolute.size else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = absolute / np.abs(desired_array)
    finite = relative[np.isfinite(relative)]
    max_relative = float(np.max(finite)) if finite.size else 0.0
    return max_absolute, max_relative


@dataclasses.dataclass(frozen=True)
class Measurement:
    nodeid: str
    backend: str
    experiment: str
    field: str
    max_abs: float
    max_rel: float
    atol: float
    rtol: float


@dataclasses.dataclass(frozen=True)
class DriftWarning:
    nodeid: str
    field: str
    atol: float
    max_abs: float


class ToleranceRecorder:
    """Collects tolerance measurements and drift warnings for the current test session."""

    def __init__(self) -> None:
        self.measurements: list[Measurement] = []
        self.drift_warnings: list[DriftWarning] = []
        self._nodeid = ""
        self._backend = ""
        self._experiment = ""

    def set_context(self, *, nodeid: str, backend: str, experiment: str) -> None:
        self._nodeid = nodeid
        self._backend = backend
        self._experiment = experiment

    def record_measurement(
        self, *, field: str, atol: float, rtol: float, max_abs: float, max_rel: float
    ) -> None:
        self.measurements.append(
            Measurement(
                nodeid=self._nodeid,
                backend=self._backend,
                experiment=self._experiment,
                field=field,
                max_abs=max_abs,
                max_rel=max_rel,
                atol=atol,
                rtol=rtol,
            )
        )

    def record_drift(self, *, field: str, atol: float, max_abs: float) -> None:
        self.drift_warnings.append(
            DriftWarning(nodeid=self._nodeid, field=field, atol=atol, max_abs=max_abs)
        )

    def dump(self, path: pathlib.Path) -> None:
        """Write the collected measurements to 'path' as JSON lines."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for measurement in self.measurements:
                stream.write(json.dumps(dataclasses.asdict(measurement)) + "\n")


class _RecorderState:
    """Holds the process-wide active recorder (avoids a module-level 'global')."""

    active: ToleranceRecorder | None = None


def get_active_recorder() -> ToleranceRecorder | None:
    return _RecorderState.active


def activate_recorder() -> ToleranceRecorder:
    _RecorderState.active = ToleranceRecorder()
    return _RecorderState.active


def deactivate_recorder() -> None:
    _RecorderState.active = None


def load_dimension_keyed_tolerances(path: pathlib.Path) -> dict[gtx.Dimension, dict[str, float]]:
    """
    Load a tolerance table keyed by horizontal dimension and experiment name.

    The JSON file maps a dimension name ('Cell', 'Edge', 'Vertex') to a mapping of experiment name
    to absolute tolerance. The returned dictionary is keyed by the corresponding 'Dimension' object
    so it can be indexed as 'table[dims.CellDim][experiment_name]'.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return {getattr(dims, f"{name}Dim"): tolerances for name, tolerances in data.items()}


def load_measurements(paths: collections.abc.Iterable[pathlib.Path]) -> list[Measurement]:
    """Load recorded measurements from one or more JSON-lines files."""
    return [
        Measurement(**json.loads(line))
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def aggregate_max_abs(
    measurements: collections.abc.Iterable[Measurement],
    *,
    backends: collections.abc.Container[str] = DETERMINISTIC_CPU_BACKENDS,
) -> dict[tuple[str, str], float]:
    """
    Reduce measurements to the maximum absolute difference per '(field, experiment)'.

    Only measurements from 'backends' are considered, so tolerances are derived exclusively from
    deterministic backends by default.
    """
    aggregated: dict[tuple[str, str], float] = {}
    for measurement in measurements:
        if measurement.backend not in backends:
            continue
        key = (measurement.field, measurement.experiment)
        aggregated[key] = max(aggregated.get(key, 0.0), measurement.max_abs)
    return aggregated


def propose_tolerance(max_abs: float) -> float:
    """
    Propose a tolerance from a measured maximum difference.

    The measured difference is scaled by 'SAFETY_FACTOR' and rounded up to one significant figure.
    A measured difference of zero yields an exact (zero) tolerance.
    """
    scaled = max_abs * SAFETY_FACTOR
    if scaled <= 0.0:
        return 0.0
    exponent = math.floor(math.log10(scaled))
    fraction = scaled / 10.0**exponent
    return math.ceil(fraction) * 10.0**exponent
