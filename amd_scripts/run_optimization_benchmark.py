#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay the validated dycore experiments without changing the review checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


EXPERIMENT_COMMIT = "cdc034acb"
DEPENDENCIES = {
    "gt4py": "eb763b97515a76c70e60befcba44dcf3bd18fb65",
    "dace": "5115128a73dc518071dbe9580b63d382540efe46",
}
BUNDLES = {
    "theta": "theta_shared_timing",
    "solver-increment": "solver_scan_fusion",
    "combined": "combined_fusion",
}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def check_comparison(platform: str, comparison: str) -> None:
    if platform == "nvidia" and comparison != "combined":
        raise ValueError("The archived separate comparisons are AMD-only; use combined on NVIDIA.")


def dependency_roots() -> dict[str, Path]:
    roots = {}
    for name, expected in DEPENDENCIES.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"Cannot locate {name}; activate the configured GPU environment.")
        root = Path(git(Path(spec.origin).parent, "rev-parse", "--show-toplevel"))
        if git(root, "rev-parse", "HEAD") != expected:
            raise ValueError(f"{name} must use the measured revision {expected} for this replay.")
        roots[name] = root
    return roots


def prepare_snapshot(root: Path, out: Path, dependencies: dict[str, Path]) -> Path:
    try:
        commit = git(root, "rev-parse", "--verify", f"{EXPERIMENT_COMMIT}^{{commit}}")
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "The pinned experiment commit is unavailable. Fetch the mi300_opt branch "
            "from your fork before submitting this job."
        ) from error
    out.mkdir(parents=True, exist_ok=False)
    stack = out / "snapshot"
    stack.mkdir()
    snapshot = stack / "icon4py"
    subprocess.run(
        ["git", "-C", str(root), "worktree", "add", "--detach", str(snapshot), commit],
        check=True,
    )
    for name, source in dependencies.items():
        (stack / name).symlink_to(source, target_is_directory=True)
    if (root / "testdata").exists() and not (snapshot / "testdata").exists():
        (snapshot / "testdata").symlink_to((root / "testdata").resolve(), target_is_directory=True)
    (out / "REPLAY.json").write_text(
        json.dumps(
            {
                "review_commit": git(root, "rev-parse", "HEAD"),
                "experiment_commit": commit,
                "snapshot": str(snapshot),
                "dependencies": {name: str(path) for name, path in dependencies.items()},
                "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "Pinned experiment replay matching the mi300_opt-based review; later local edits are not included.",
            },
            indent=2,
        )
        + "\n"
    )
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("amd", "nvidia"))
    parser.add_argument("--comparison", default="combined", choices=tuple(BUNDLES))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    check_comparison(args.platform, args.comparison)
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    snapshot = prepare_snapshot(root, out, dependency_roots())
    bundle = snapshot / "amd_scripts" / BUNDLES[args.comparison]
    sources = sorted(
        p.parent / "src"
        for p in (snapshot / "model").rglob("pyproject.toml")
        if (p.parent / "src").is_dir()
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(bundle), *map(str, sources), env.get("PYTHONPATH", "")]
    )
    # The archived theta wrapper also compared the historical Python rewrite.
    # Select only its native comparison; all model/compiler/timing code stays pinned.
    worker = """import os, sys
from pathlib import Path
from icon4py.model.common import dimension
assert Path(dimension.__file__).resolve().is_relative_to(Path.cwd()), "Wrong model checkout imported"
import measure
if os.environ["DYCORE_REPLAY_COMPARISON"] == "theta":
    comparisons = measure.comparisons
    measure.comparisons = lambda: comparisons()[:1]
measure.main()
"""
    env["DYCORE_REPLAY_COMPARISON"] = args.comparison
    command = [sys.executable, "-c", worker, "--output", str(out / "results")]
    if args.comparison == "combined":
        command += ["--platform", args.platform]
    subprocess.run(command, cwd=snapshot, env=env, check=True)
    print(f"Results: {out / 'results' / 'TIMING_SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
