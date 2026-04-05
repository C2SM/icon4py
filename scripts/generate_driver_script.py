#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a SLURM submission script for the icon4py standalone driver."""

import argparse
import math
import re
import textwrap
from pathlib import Path


GPUS_PER_NODE = 4

# Recommended number of ranks (GPUs) per grid resolution.
DEFAULT_RANKS = {
    (2, 4): 1,
    (2, 5): 1,
    (2, 6): 2,
    (2, 7): 8,
    (2, 8): 32,
    (2, 9): 128,
    (2, 10): 512,
    (2, 11): 2048,
}

GRID_DIR = "/capstor/store/cscs/userlab/cwd01/cong/grids"

# Default grid file per resolution (picking one when duplicates exist).
DEFAULT_GRID_FILES = {
    (2, 4): f"{GRID_DIR}/icon_grid_0010_R02B04_G.nc",
    (2, 5): f"{GRID_DIR}/icon_grid_0008_R02B05_G.nc",
    (2, 6): f"{GRID_DIR}/icon_grid_0002_R02B06_G.nc",
    (2, 7): f"{GRID_DIR}/icon_grid_0004_R02B07_G.nc",
    (2, 8): f"{GRID_DIR}/icon_grid_0033_R02B08_G.nc",
    (2, 9): f"{GRID_DIR}/icon_grid_0015_R02B09_G.nc",
    (2, 10): f"{GRID_DIR}/icon_grid_0017_R02B10_G.nc",
}

# Recommended dtime (seconds) per resolution.
DEFAULT_DTIME = {
    (2, 4): 360.0,
    (2, 5): 180.0,
    (2, 6): 75.0,
    (2, 7): 40.0,
    (2, 8): 20.0,
    (2, 9): 10.0,
    (2, 10): 5.0,
}


def parse_resolution(grid_file: str) -> tuple[int, int] | None:
    """Try to extract (root, bisection) from the grid file name, e.g. R02B09."""
    m = re.search(r"R(\d+)B(\d+)", grid_file, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def get_default_ranks(root: int, bisection: int) -> int:
    key = (root, bisection)
    if key in DEFAULT_RANKS:
        return DEFAULT_RANKS[key]
    # Fallback: scale from r2b8 baseline (8 ranks for ~5.2M cells)
    num_cells = 20 * root**2 * 4**bisection
    return max(1, round(num_cells / 650_000))


RUN_WRAPPER = textwrap.dedent("""\
    #!/bin/bash
    # GPU pinning and NUMA binding wrapper for Santis (4 GPUs per node).
    export LOCAL_RANK=$SLURM_LOCALID
    export GLOBAL_RANK=$SLURM_PROCID

    GPUS=(0 1 2 3)
    NUMA=(0 1 2 3)
    NUMA_NODE=${NUMA[$LOCAL_RANK]}

    export CUDA_VISIBLE_DEVICES=${GPUS[$LOCAL_RANK]}

    ulimit -s unlimited
    numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE "$@"
""")


def generate_script(  # noqa: PLR0912
    grid_file: str | None,
    ranks: int | None,
    root: int | None,
    bisection: int | None,
    time: str,
    account: str,
    backend: str,
    venv: str | None,
    log_level: str,
    dtime: float | None = None,
    output_frequency: str = "none",
    uenv: bool = True,
) -> tuple[str, str, str]:
    """Return (slurm_script, run_wrapper, job_name) strings."""
    # Resolve resolution: from explicit args, grid file name, or defaults
    if root is not None and bisection is not None:
        res_root, res_bisection = root, bisection
    elif grid_file is not None:
        parsed = parse_resolution(grid_file)
        if parsed is None:
            raise SystemExit(
                "Cannot determine grid resolution from filename. "
                "Please specify --root and --bisection explicitly."
            )
        res_root, res_bisection = parsed
    else:
        raise SystemExit("Please specify either --grid-file or --root and --bisection.")

    res_tag = f"r{res_root}b{res_bisection}"

    # Resolve grid file from lookup if not given explicitly
    if grid_file is None:
        key = (res_root, res_bisection)
        if key not in DEFAULT_GRID_FILES:
            raise SystemExit(f"No default grid file for {res_tag}. Please specify --grid-file.")
        grid_file = DEFAULT_GRID_FILES[key]

    # Resolve dtime from lookup if not given explicitly
    if dtime is None:
        key = (res_root, res_bisection)
        dtime = DEFAULT_DTIME.get(key, 10.0)

    # Resolve rank count
    if ranks is None:
        ranks = get_default_ranks(res_root, res_bisection)

    nodes = math.ceil(ranks / GPUS_PER_NODE)
    ntasks_per_node = min(ranks, GPUS_PER_NODE)

    job_name = f"icon4py_{res_tag}_{ranks}ranks"
    run_dir = str(Path.cwd() / job_name)

    # Resolve venv: prefer cwd/.venv, then repo/.venv
    if venv is None:
        cwd_venv = Path.cwd() / ".venv"
        repo_venv = Path(__file__).resolve().parent.parent / ".venv"
        if cwd_venv.is_dir():
            venv = str(cwd_venv.resolve())
        elif repo_venv.is_dir():
            venv = str(repo_venv)
        else:
            raise SystemExit(
                "No .venv found in the current directory or repo root. "
                "Please specify --venv explicitly."
            )

    # Resolve main.py path relative to repo
    repo_root = Path(__file__).resolve().parent.parent
    main_py = (
        repo_root
        / "model"
        / "standalone_driver"
        / "src"
        / "icon4py"
        / "model"
        / "standalone_driver"
        / "main.py"
    )

    uenv_lines = ""
    if uenv:
        uenv_lines = "#SBATCH --uenv=icon/25.2:v3:/user-environment\n#SBATCH --view=default\n"

    script = textwrap.dedent(f"""\
        #! /bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --output={run_dir}/log_stdout_%j.txt
        #SBATCH --error={run_dir}/log_%j.txt
        #SBATCH --account={account}
        {uenv_lines}#SBATCH --nodes={nodes}
        #SBATCH --ntasks-per-node={ntasks_per_node}
        #SBATCH --partition=normal
        #SBATCH --time={time}

        set -euo pipefail

        JOB_DIR={run_dir}
        mkdir -p "${{JOB_DIR}}"

        source {venv}/bin/activate

        export GT4PY_BUILD_CACHE_LIFETIME=PERSISTENT
        export GT4PY_BUILD_CACHE_DIR={Path.cwd()}/gt4py_cache
        export LD_LIBRARY_PATH=${{LD_LIBRARY_PATH:-}}:/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nvhpc-25.1-tsfur7lqj6njogdqafhpmj5dqltish7t/Linux_aarch64/25.1/compilers/lib
        export CC=$(which gcc)
        export CXX=$(which g++)
        export MPICH_CC=$(which gcc)
        export MPICH_CXX=$(which g++)
        export MPICH_GPU_SUPPORT_ENABLED=1
        export FI_CXI_RX_MATCH_MODE=software
        export FI_MR_CACHE_MONITOR=disabled
        export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=0
        export CUPY_CACHE_IN_MEMORY=1
        export PYTHONOPTIMIZE=1

        export OUTPUT_PATH=${{JOB_DIR}}/output
        export INPUT_GRID={grid_file}

        srun "${{JOB_DIR}}/run_wrapper.sh" python {main_py} \\
            --output-path ${{OUTPUT_PATH}} \\
            --grid-file-path ${{INPUT_GRID}} \\
            --icon4py-backend {backend} \\
            --log-level {log_level} \\
            --dtime {dtime} \\
            --output-frequency {output_frequency}
    """)
    return script, RUN_WRAPPER, job_name


def main():
    parser = argparse.ArgumentParser(
        description="Generate a SLURM submission script for the icon4py standalone driver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--grid-file",
        default=None,
        help="Path to the ICON grid file. Looked up from resolution if not set.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=None,
        help="Number of MPI ranks (GPUs). Auto-scaled from grid resolution if not set.",
    )
    parser.add_argument("--root", type=int, default=None, help="Grid root (e.g. 2).")
    parser.add_argument(
        "--bisection", type=int, default=None, help="Grid bisection level (e.g. 9)."
    )
    parser.add_argument("--time", default="2:00:00", help="SLURM wall-clock time limit.")
    parser.add_argument("--account", default="cwd01", help="SLURM account.")
    parser.add_argument(
        "--no-uenv",
        action="store_true",
        help="Disable #SBATCH --uenv and --view directives.",
    )
    parser.add_argument("--backend", default="dace_gpu", help="GT4Py backend.")
    parser.add_argument(
        "--venv",
        default=None,
        help="Path to the virtual environment (defaults to <repo>/.venv).",
    )
    parser.add_argument("--log-level", default="warning", help="Python log level.")
    parser.add_argument(
        "--dtime",
        type=float,
        default=None,
        help="Time step in seconds. Auto-selected from resolution if not set "
        "(r2b6=75, r2b7=40, r2b8=20, r2b9=10, r2b10=5).",
    )
    parser.add_argument(
        "--output-frequency",
        choices=["none", "hourly", "daily", "final"],
        default="none",
        help="Diagnostic output frequency.",
    )
    args = parser.parse_args()

    script, wrapper, job_name = generate_script(
        grid_file=args.grid_file,
        ranks=args.ranks,
        root=args.root,
        bisection=args.bisection,
        time=args.time,
        account=args.account,
        backend=args.backend,
        venv=args.venv,
        log_level=args.log_level,
        dtime=args.dtime,
        output_frequency=args.output_frequency,
        uenv=not args.no_uenv,
    )

    out_dir = Path.cwd() / job_name
    if out_dir.exists():
        answer = input(f"{out_dir}/ already exists. Override? [y/N] ").strip().lower()
        if answer != "y":
            raise SystemExit("Aborted.")

    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = out_dir / f"{job_name}.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)

    wrapper_path = out_dir / "run_wrapper.sh"
    wrapper_path.write_text(wrapper)
    wrapper_path.chmod(0o755)

    print(f"Written to {out_dir}/")


if __name__ == "__main__":
    main()
