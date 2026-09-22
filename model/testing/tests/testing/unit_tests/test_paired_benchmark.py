# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from dataclasses import dataclass

import numpy as np
import pytest

from icon4py.model.testing import paired_benchmark as paired


def test_schedule_matches_archived_order():
    schedule = paired.timing_schedule()
    assert len(schedule) == 24
    assert schedule[:2] == [("intervention", 0, "BAAB"), ("placebo", 0, "BAAB")]
    for quartet in range(12):
        phases = [x for x in schedule if x[1] == quartet]
        assert {x[0] for x in phases} == {"intervention", "placebo"}
        assert phases[0][2] == phases[1][2]
    assert sum(x[2] == "ABBA" for x in schedule) == 12


def test_control_bias_prevents_claim():
    intervention, controls = [], []
    for phase, _quartet, order in paired.timing_schedule():
        target = controls if phase == "placebo" else intervention
        for arm in order:
            target.append(
                dict(arm=arm, value=10.0 - (arm == "B") * (0.3 if phase == "placebo" else 0.1))
            )
    result = paired.summarize(intervention, controls, lambda b: b["value"])
    assert result["saved_ms"] == pytest.approx(0.1)
    assert result["control_threshold_ms"] == pytest.approx(0.3)
    assert result["status"] == "unresolved"


@dataclass
class Model:
    field: np.ndarray
    cached_timestep: float = 0.0


Model.__module__ = "icon4py.model.testing.test_paired_benchmark"


def test_snapshot_restores_cached_scalar_and_views():
    array = np.arange(8.0)
    model = Model(array[::2])
    snap = paired.ModelSnapshot([model, array], lambda: None)
    model.cached_timestep = 90.0
    array[:] = -1
    snap.restore()
    assert model.cached_timestep == 0.0
    np.testing.assert_array_equal(array, np.arange(8.0))
    model.new_cached_scalar = 1.0
    with pytest.raises(ValueError, match="inventory"):
        snap.restore()


def test_snapshot_rejects_wrong_results():
    model = Model(np.ones(2))
    snap = paired.ModelSnapshot([model], lambda: None)
    reference = [a.copy() for a in snap.saved]
    model.field[0] = 2
    with pytest.raises(AssertionError):
        snap.validate(reference)


def test_real_pytest_timer_keeps_setup_outside_samples(benchmark, monkeypatch):
    """Exercise the actual pedantic API, including stats and JSON extra_info."""
    model = Model(np.ones(2))
    model.max_vertical_cfl = np.array(0.0)
    selected, observed, offsets = ["A"], [], []
    monkeypatch.setattr(paired, "provenance", lambda paths: {})
    monkeypatch.setattr(paired, "_metric_offsets", lambda: offsets.append(0))
    monkeypatch.setattr(
        paired, "_program_samples", lambda before, metric_name="compute": {"program": 1.0}
    )

    def select(arm):
        selected[0] = arm

    def callback():
        observed.append((selected[0], model.cached_timestep))
        assert float(model.max_vertical_cfl) == model.cached_timestep
        model.max_vertical_cfl = np.asarray(model.max_vertical_cfl + 1.0)
        model.field[:] += 1
        model.cached_timestep += 1

    paired.compare(
        benchmark,
        callback,
        select=select,
        roots=[model],
        synchronize=lambda: None,
        metadata={},
        source_files=[],
        samples_per_block=2,
        warmup=1,
    )
    report = benchmark.extra_info["comparison"]
    assert report["validation"]["max_abs_error"] == 0
    assert len(report["blocks"]) == 96
    assert len(benchmark.stats.stats.data) == len(offsets) == 192
    assert len(observed) == 2 + 96 * 3
    assert [
        sample["wall_ms"] for block in report["blocks"] for sample in block["samples"]
    ] == pytest.approx([value * 1000 for value in benchmark.stats.stats.data])
    for index, block in enumerate(report["blocks"]):
        calls = observed[2 + index * 3 : 2 + (index + 1) * 3]
        arm = "A" if block["phase"] == "placebo" else block["arm"]
        assert calls == [(arm, 0.0), (arm, 1.0), (arm, 2.0)]
        assert all(sample["wall_ms"] >= 0 for sample in block["samples"])
    assert model.cached_timestep == 0.0
    np.testing.assert_array_equal(model.field, np.ones(2))


@dataclass
class CompilerObject:
    workspace: np.ndarray

    def __call__(self, **kwargs):
        pass


def test_snapshot_does_not_follow_compiler_workspaces():
    model = Model(np.ones(2))
    model.program = functools.partial(CompilerObject(np.zeros(3)), field=np.ones(4))
    snap = paired.ModelSnapshot([model], lambda: None)
    assert sorted(a.size for a in snap.arrays.values()) == [2, 4]


@pytest.mark.parametrize("replace_array", [False, True])
def test_snapshot_restores_and_validates_scalar_arrays(replace_array):
    model = Model(np.ones(2))
    original = model.max_vertical_cfl = np.array(0.0)
    snap = paired.ModelSnapshot([model], lambda: None)
    reference = [a.copy() for a in snap.current_arrays()]
    if replace_array:
        model.max_vertical_cfl = np.asarray(np.maximum(model.max_vertical_cfl, 3.0))
    else:
        model.max_vertical_cfl[...] = 3.0
    snap.verify_inventory()
    with pytest.raises(AssertionError):
        snap.validate(reference)
    model.cached_timestep = 90.0
    snap.restore()
    assert model.max_vertical_cfl is original
    assert float(model.max_vertical_cfl) == 0.0
    assert model.cached_timestep == 0.0
    assert snap.validate(reference)["max_abs_error"] == 0.0
    model.max_vertical_cfl = np.zeros(2)
    with pytest.raises(ValueError, match="inventory"):
        snap.restore()


def test_snapshot_still_rejects_replaced_field_buffers():
    model = Model(np.ones(2))
    snap = paired.ModelSnapshot([model], lambda: None)
    model.field = model.field.copy()
    with pytest.raises(ValueError, match="inventory"):
        snap.restore()
