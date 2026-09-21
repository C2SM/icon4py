# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: PLC0415
"""Run controlled comparisons from the two published PR checkouts."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from fusion_core import COMPARISONS, TARGET, analyze, require, save_json, timing_schedule


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def source_hashes(roots):
    result = {}
    for label, root in roots.items():
        result[label] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))
            if not {"__pycache__", ".gt4py_cache", ".dacecache"}.intersection(path.parts)
        }
    return result


def preflight(platform):
    """Reject another editable checkout, compiler overlay, or dependency revision."""
    import cupy as cp
    import dace
    import gt4py
    from gt4py.next.program_processors.runners.dace import scan_fusion
    from gt4py.next.program_processors.runners.dace.transformations import map_fusion_extended

    stack = json.loads((HERE / "stack.json").read_text())
    gt_root = Path(gt4py.__file__).resolve().parents[2]
    require((gt_root / ".git").exists(), "Install the companion GT4Py checkout editable.")
    require(
        git(gt_root, "rev-parse", "HEAD") == stack["gt4py_commit"],
        "GT4Py revision differs from stack.json; use the pinned companion PR commit.",
    )
    require(
        not git(gt_root, "status", "--porcelain", "--untracked-files=all", "--", "src"),
        "GT4Py has uncommitted source changes.",
    )
    roots = dict(
        gt4py=gt_root / "src",
        icon4py=REPO / "model",
        harness=HERE,
        dace=Path(dace.__file__).resolve().parent,
    )
    for module in (
        "common.model_options",
        "atmosphere.dycore.solve_nonhydro",
        "testing.fixtures.stencil_tests",
        "driver",
    ):
        mod = importlib.import_module("icon4py.model." + module)
        require(
            Path(mod.__file__).resolve().is_relative_to(REPO / "model"),
            f"Wrong ICON4Py checkout for {module}.",
        )
    for module in (scan_fusion, map_fusion_extended):
        require(
            Path(module.__file__).resolve().is_relative_to(roots["gt4py"]),
            "Compiler overlay detected.",
        )
    require(
        callable(scan_fusion.normalize_scan_producers) and callable(scan_fusion.fuse_scan_inputs),
        "Missing scan fusion support.",
    )
    require(
        "allow_shared_data" in map_fusion_extended.VerticalSplitMapRange.__properties__,
        "Missing shared-output fusion support.",
    )
    direct = json.loads(
        importlib.metadata.distribution("dace").read_text("direct_url.json") or "{}"
    )
    revision = direct.get("vcs_info", {}).get("commit_id")
    if revision is None:
        dace_root = Path(dace.__file__).resolve().parents[1]
        require((dace_root / ".git").exists(), "Install the pinned DaCe Git revision.")
        revision = git(dace_root, "rev-parse", "HEAD")
        require(
            not git(dace_root, "status", "--porcelain", "--untracked-files=all", "--", "dace"),
            "DaCe source has local changes.",
        )
    require(revision == stack["dace_commit"], "DaCe revision differs from stack.json.")
    require(
        bool(cp.cuda.runtime.is_hip) == (platform == "amd"),
        "CuPy runtime does not match the selected vendor.",
    )
    require(cp.cuda.runtime.getDeviceCount() >= 1, "No visible GPU.")
    require(
        float(cp.arange(16, dtype=cp.float64).sum()) == 120.0,
        "CuPy JIT/reduction preflight failed.",
    )
    require(
        not git(
            REPO,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "model",
            "amd_scripts/fusion_benchmark",
        ),
        "ICON4Py/harness source has uncommitted changes.",
    )
    return roots, dict(
        icon4py_commit=git(REPO, "rev-parse", "HEAD"),
        gt4py_commit=stack["gt4py_commit"],
        dace_commit=revision,
        source_roots={k: str(v) for k, v in roots.items()},
        python=sys.version,
        packages=sorted(
            f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
        ),
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                ("DACE_", "GT4PY_", "ICON4PY_", "CUDA_", "HIP", "ROCM", "SLURM_JOB", "SLURM_NODE")
            )
            or k in ("CC", "CXX", "CUPY_ACCELERATORS", "PYTHONHASHSEED", "PYTHONOPTIMIZE")
        },
        gpu_properties=cp.cuda.runtime.getDeviceProperties(0),
    )


def cmake_command(arguments):
    """Recover inherited SIGCHLD masking and time out the whole compiler process group."""
    previous = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    print(f"CMake wrapper: inherited SIGCHLD blocked={signal.SIGCHLD in previous}", flush=True)
    proc = subprocess.Popen([os.environ["FUSION_REAL_CMAKE"], *arguments], start_new_session=True)
    try:
        return proc.wait(timeout=int(os.environ.get("FUSION_CMAKE_TIMEOUT", "1200")))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return 124


def configure_cmake(output):
    real = shutil.which("cmake")
    require(real, "CMake is missing from the activated environment.")
    tools = output / "compiler-tools"
    tools.mkdir()
    wrapper = tools / "cmake"
    # Python's interpreter path is embedded as a Python literal, not shell code.
    wrapper.write_text(
        "#!/usr/bin/env python3\nimport os\nos.execv("
        + repr(sys.executable)
        + ", "
        + repr([sys.executable, str(Path(__file__).resolve()), "--cmake"])
        + " + __import__('sys').argv[1:])\n"
    )
    wrapper.chmod(0o755)
    os.environ.update(FUSION_REAL_CMAKE=real, PATH=str(tools) + os.pathsep + os.environ["PATH"])
    project = output / "cmake-probe"
    project.mkdir()
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.15)\nproject(probe LANGUAGES CXX)\nadd_executable(probe main.cpp)\n"
    )
    (project / "main.cpp").write_text("int main() { return 0; }\n")
    env = dict(os.environ, FUSION_CMAKE_TIMEOUT="120")
    for args in (
        ["-S", str(project), "-B", str(project / "build"), "-G", "Ninja"],
        ["--build", str(project / "build"), "--parallel", "1"],
    ):
        subprocess.run([str(wrapper), *args], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("amd", "nvidia"))
    parser.add_argument(
        "--grids", nargs="+", choices=("regional", "global"), default=["regional", "global"]
    )
    parser.add_argument("--comparisons", nargs="+", choices=COMPARISONS, default=["combined"])
    parser.add_argument("--quartets", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the pinned GPU environment without compiling/running the granule.",
    )
    args = parser.parse_args()
    timing_schedule(args.quartets, 20260915)
    require(args.warmup >= 1 and args.rounds >= 1, "Warmup and rounds must be positive.")
    require(
        len(args.grids) == len(set(args.grids))
        and len(args.comparisons) == len(set(args.comparisons)),
        "Duplicate grids or comparisons.",
    )
    os.chdir(REPO)
    os.environ.update(
        ICON4PY_DACE_THETA_FUSION="0",
        ICON4PY_DACE_SOLVER_FUSION="0",
        GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1",
        GT4PY_BUILD_CACHE_LIFETIME="persistent",
        DACE_compiler_build_folder_mode="development",
        GT4PY_COLLECT_METRICS_LEVEL="10",
        GT4PY_ADD_GPU_TRACE_MARKERS="1",
        GT4PY_BUILD_JOBS="0",
        GT4PY_BUILD_JOBS_MODE="serial",
        PYTHONOPTIMIZE="2",
        PYTHONHASHSEED="0",
        PYTEST_ADDOPTS="",
    )
    os.environ.setdefault("ICON4PY_BACKEND_WORKSPACE_SIZE", "8589934592")
    if args.platform == "amd":
        os.environ.setdefault("DACE_compiler_cuda_chiplet_number", "1")
    roots, manifest = preflight(args.platform)
    if args.check:
        print(json.dumps(manifest, indent=2, default=str))
        return
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output
        or REPO / "fusion-results" / f"{args.platform}_{os.environ.get('SLURM_JOB_ID', stamp)}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = source_hashes(roots)
    save_json(output / "manifest.json", manifest)
    save_json(output / "source_hashes.json", before)
    results = {}
    try:
        configure_cmake(output)
        for grid in args.grids:
            fingerprints = []
            for comparison in args.comparisons:
                destination = output / grid / comparison
                destination.mkdir(parents=True)
                spec = dict(
                    name=comparison,
                    platform=args.platform,
                    warmup=args.warmup,
                    rounds=args.rounds,
                    quartets=args.quartets,
                    input_seed=20260910,
                    order_seed=20260915,
                )
                save_json(destination / "case.json", spec)
                env = dict(
                    os.environ,
                    FUSION_CASE=json.dumps(spec),
                    FUSION_OUTPUT=str(destination),
                    GT4PY_BUILD_CACHE_DIR=str(destination / "build"),
                    GT4PY_METRICS_OUTPUT_PATH=str(destination / "program-metrics.json"),
                )
                # Only the pytest adapter is added; never a compiler or model overlay.
                env["PYTHONPATH"] = str(HERE) + (
                    os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
                )
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-sv",
                    "-p",
                    "fusion_plugin",
                    "-p",
                    "no:tach",
                    "-m",
                    "continuous_benchmarking",
                    "--benchmark-disable",
                    "--backend=dace_gpu",
                    f"--grid=icon_benchmark_{grid}:120",
                    TARGET,
                ]
                save_json(destination / "command.json", command)
                print(
                    f"Running {grid}/{comparison}; progress: {destination / 'run.log'}", flush=True
                )
                with (destination / "run.log").open("w") as log:
                    subprocess.run(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                report = json.loads((destination / "timing.json").read_text())
                results[f"{grid}/{comparison}"] = analyze(report)
                fingerprints.append(
                    (
                        report["initial_state_sha256"],
                        report["initial_scalar_state_sha256"],
                        report["grid_dimensions"],
                    )
                )
                require(
                    all(f == fingerprints[0] for f in fingerprints),
                    "Comparisons used different initial state or grids.",
                )
                save_json(output / "RESULTS.json", results)
        after = source_hashes(roots)
        save_json(output / "source_hashes.final.json", after)
        require(before == after, "Sources changed during the run; timings are not reviewable.")
        lines = [
            "# Controlled fusion results",
            "",
            "Times are per solve_nonhydro granule call. Positive saving means faster.",
            "",
            "| Grid / comparison | Metric | Original ms | Optimised ms | Saving | Assessment |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
        for name, result in results.items():
            for metric in ("device", "wall"):
                r = result[metric]
                lines.append(
                    f"| {name} | {metric} | {r['baseline_ms']:.6f} | {r['variant_ms']:.6f} | {r['saving_percent']:.2f}% | {r['status']} |"
                )
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
        (output / "COMPLETE").write_text(
            "Correctness, timing, provenance and source checks passed. Performance status is in RESULTS.md.\n"
        )
        print((output / "RESULTS.md").read_text())
    except BaseException as error:
        save_json(output / "FAILED.json", dict(error=f"{type(error).__name__}: {error}"))
        raise


if __name__ == "__main__":
    if sys.argv[1:2] == ["--cmake"]:
        sys.exit(cmake_command(sys.argv[2:]))
    main()
