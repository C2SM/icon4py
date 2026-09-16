# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Local safety checks for the isolated benchmark launcher; no GPU required."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "optimization_replay", Path(__file__).parents[1] / "run_optimization_benchmark.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_replay_uses_pinned_source_without_changing_checkout(tmp_path, monkeypatch):
    root = tmp_path / "review"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    source = root / "model.py"
    source.write_text("pinned model\n")
    replay.git(root, "add", "model.py")
    replay.git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "Pinned model",
    )
    commit = replay.git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(replay, "EXPERIMENT_COMMIT", commit)
    source.write_text("review changes must survive\n")
    (root / "testdata").mkdir()
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    out = tmp_path / "run"
    snapshot = replay.prepare_snapshot(root, out, {"gt4py": dependency})
    assert (snapshot / "model.py").read_text() == "pinned model\n"
    assert source.read_text() == "review changes must survive\n"
    assert replay.git(root, "rev-parse", "HEAD") == commit
    assert (snapshot.parent / "gt4py").resolve() == dependency
    assert (snapshot / "testdata").resolve() == root / "testdata"
    assert json.loads((out / "REPLAY.json").read_text())["experiment_commit"] == commit
    with pytest.raises(FileExistsError):
        replay.prepare_snapshot(root, out, {})


@pytest.mark.parametrize("comparison", ["theta", "solver-increment"])
def test_rejects_amd_only_comparisons_on_nvidia(comparison):
    with pytest.raises(ValueError, match="AMD-only"):
        replay.check_comparison("nvidia", comparison)
    replay.check_comparison("amd", comparison)


def test_combined_supported_on_both_platforms():
    for platform in ("amd", "nvidia"):
        replay.check_comparison(platform, "combined")
