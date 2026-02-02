# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Run serialization jobs, collect ser_data and NAMELISTS, and archive outputs."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import typer

from icon4py.model.testing import datatest_utils as dt_utils, definitions


cli = typer.Typer(no_args_is_help=True, help=__doc__)


# ======================================
# USER CONFIGURATION
# ======================================
COMM_SIZES: list[int] = [1]  # , 2, 4]

EXPERIMENTS = [
    # definitions.Experiments.MCH_CH_R04B09,
    # definitions.Experiments.JW,
    definitions.Experiments.EXCLAIM_APE,
    # definitions.Experiments.GAUSS3D,
    # definitions.Experiments.WEISMAN_KLEMP_TORUS,
]

# Slurm settings
SBATCH_PARTITION = "normal"
SBATCH_TIME = "00:15:00"
SBATCH_ACCOUNT = "cwd01"
SBATCH_UENV = "icon/25.2:v3"
SBATCH_UENV_VIEW = "default"
JOB_POLL_SECONDS = 10

# Base directories (adjust if needed)
PROJECTS_DIR = Path(os.environ.get("SCRATCH", str(Path.home() / "projects")))
ICONF90_DIR = PROJECTS_DIR / "icon-exclaim.serialize"
ICONF90_BUILD_FOLDER = "build_serialize"

# Derived paths
BUILD_DIR = ICONF90_DIR / ICONF90_BUILD_FOLDER
RUNSCRIPTS_DIR = BUILD_DIR / "run"
EXPERIMENTS_DIR = BUILD_DIR / "experiments"

# Output location for copied ser_data and tarballs
OUTPUT_ROOT = EXPERIMENTS_DIR / definitions.SERIALIZED_DATA_DIR

# Maximum concurrent threads for running experiments
MAX_THREADS: int = 5

# ======================================
# END USER CONFIGURATION
# ======================================


def get_f90exp_name(experiment: definitions.Experiment) -> str:
    return f"{experiment.name}_sb"


def get_f90exp_dir(experiment: definitions.Experiment) -> Path:
    return EXPERIMENTS_DIR / get_f90exp_name(experiment)


def get_nmlfile_name(experiment: definitions.Experiment) -> str:
    return f"exp.{get_f90exp_name(experiment)}"


def get_slurmscript_name(experiment: definitions.Experiment) -> str:
    return f"{get_nmlfile_name(experiment)}.run"


def get_serdata_dst_dir(experiment: definitions.Experiment, comm_size: int) -> Path:
    """Get the destination directory for serialized data."""
    return OUTPUT_ROOT / dt_utils.get_ranked_experiment_name_with_version(experiment, comm_size)


def get_tar_path(experiment: definitions.Experiment, comm_size: int) -> Path:
    """Get the path to the tar archive for the experiment."""
    return OUTPUT_ROOT / dt_utils.get_experiment_archive_filename(experiment, comm_size)


def cleanup_exp_output(experiment: definitions.Experiment, comm_size: int) -> None:
    """Clean up experiment output directories and archives.

    Deletes:
    - Experiment directory (exp_dir)
    - Serialized data destination directory (dest_dir)
    - Tar archive (tar_path)
    """
    # Delete experiment directory
    exp_dir = get_f90exp_dir(experiment)
    if exp_dir.exists():
        shutil.rmtree(exp_dir)

    # Delete serialized data destination directory
    dest_dir = get_serdata_dst_dir(experiment, comm_size)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    # Delete tar archive
    tar_path = get_tar_path(experiment, comm_size)
    if tar_path.exists():
        tar_path.unlink()


def run_command(
    cmd: list[str], check: bool = True, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=True, cwd=cwd)


def log_status(message: str) -> None:
    """Log a status message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def parse_extra_mpi_ranks(script_path: Path, comm_size: int) -> int:
    """Parse extra MPI ranks from the Fortran script by summing num_* variables
    found in the &parallel_nml section.

    Looks for lines starting with:
        num_io_procs      =
        num_prefetch_proc =
        num_restart_procs =

    Supports both direct values (e.g., "num_io_procs = 1") and variable references
    (e.g., "num_io_procs = ${num_io_procs}") where the variable is defined elsewhere
    in the file.

    Args:
        script_path: Path to the script file to parse
        comm_size: Communicator size used for the run

    Returns:
        Sum of num_io_procs, num_prefetch_proc, and num_restart_procs values
    """
    content = script_path.read_text()
    extra_ranks = 0

    # First, parse all variable definitions from the entire file
    # Pattern: variable_name=value (outside of namelist sections)
    var_definitions: dict[str, int] = {}
    for match in re.finditer(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\d+)", content, flags=re.MULTILINE):
        var_name = match.group(1)
        var_value = int(match.group(2))
        var_definitions[var_name] = var_value

    # Find the &parallel_nml section
    start_match = re.search(r"^\s*&parallel_nml\b.*$", content, flags=re.MULTILINE)
    if not start_match:
        return extra_ranks

    end_match = re.search(r"^\s*/\s*$", content[start_match.end() :], flags=re.MULTILINE)
    if not end_match:
        return extra_ranks

    section_start = start_match.start()
    section_end = start_match.end() + end_match.start()
    section = content[section_start:section_end]

    # Pattern to match num_* variables with either direct values or variable references
    var_names = ["num_io_procs", "num_prefetch_proc", "num_restart_procs"]

    for var_name in var_names:
        # num_io_procs only applies to MPI runs (no extra IO ranks for serial).
        if var_name == "num_io_procs" and comm_size <= 1:
            continue
        # Try to match direct integer value
        pattern_direct = rf"{var_name}\s*=\s*(\d+)"
        match = re.search(pattern_direct, section)
        if match:
            extra_ranks += int(match.group(1))
            continue

        # Try to match variable reference ${var_name}
        pattern_ref = rf"{var_name}\s*=\s*\$\{{([a-zA-Z_][a-zA-Z0-9_]*)\}}"
        match = re.search(pattern_ref, section)
        if match:
            ref_var_name = match.group(1)
            if ref_var_name in var_definitions:
                extra_ranks += var_definitions[ref_var_name]
            # If variable not found, silently ignore (value is 0)

    return extra_ranks


def update_slurm_variables(script_path: Path) -> None:
    """Update SBATCH directives in the Slurm script (partition, account, time, uenv, view)."""
    content = script_path.read_text()

    # Find the position after #SBATCH --job-name= line
    job_name_match = re.search(r"^#SBATCH\s+--job-name=.*$", content, flags=re.MULTILINE)
    if not job_name_match:
        raise RuntimeError("Could not find #SBATCH --job-name= line in script")

    # Prepare the new SBATCH lines to insert
    new_lines = (
        f"#SBATCH --partition={SBATCH_PARTITION}\n"
        f"#SBATCH --account={SBATCH_ACCOUNT}\n"
        f"#SBATCH --time={SBATCH_TIME}\n"
        f"#SBATCH --uenv='{SBATCH_UENV}'\n"
        f"#SBATCH --view='{SBATCH_UENV_VIEW}'"
    )

    # Remove existing partition, account, time, uenv, and view lines if they exist
    content = re.sub(r"^#SBATCH\s+--partition=.*$\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"^#SBATCH\s+--account=.*$\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"^#SBATCH\s+--time=.*$\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"^#SBATCH\s+--uenv=.*$\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"^#SBATCH\s+--view=.*$\n?", "", content, flags=re.MULTILINE)

    # Re-find job-name position in the cleaned text
    job_name_match = re.search(r"^(#SBATCH\s+--job-name=.*$)", content, flags=re.MULTILINE)
    if not job_name_match:
        raise RuntimeError("Could not find #SBATCH --job-name= line in script")

    # Insert new lines after the job-name line
    insertion_point = job_name_match.end()
    content = content[:insertion_point] + "\n" + new_lines + content[insertion_point:]

    script_path.write_text(content)


def update_slurm_ranks(script_path: Path, mpi_ranks: int, extra_mpi_ranks: int = 0) -> None:
    """Update ranks in the Slurm script (ntasks-per-node and mpi_procs_pernode).

    Args:
        script_path: Path to the Slurm script
        mpi_ranks: Base number of MPI ranks
        extra_mpi_ranks: Additional ranks reserved for special operations (e.g., pre-fetch)
    """
    total_ranks = mpi_ranks + extra_mpi_ranks

    content = script_path.read_text()

    content = re.sub(
        r"^#SBATCH\s+--ntasks-per-node\s*=\s*\d+\s*$",
        f"#SBATCH --ntasks-per-node={total_ranks}",
        content,
        flags=re.MULTILINE,
    )

    content = re.sub(
        r"^:\s+\$\{no_of_nodes:=\d+\}\s+\$\{mpi_procs_pernode:=\d+\}\s*$",
        f": ${{no_of_nodes:=1}} ${{mpi_procs_pernode:={total_ranks}}}",
        content,
        flags=re.MULTILINE,
    )

    script_path.write_text(content)


def submit_job(script_path: Path) -> str:
    cmd = ["sbatch", str(script_path)]
    result = run_command(cmd, cwd=RUNSCRIPTS_DIR)
    match = re.search(r"Submitted batch job\s+(\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Unable to parse job id from sbatch output: {result.stdout}")
    return match.group(1)


def normalize_state(raw_state: str) -> str:
    cleaned = raw_state.strip().upper()
    cleaned = cleaned.split("+")[0]
    cleaned = cleaned.split(":")[0]
    return cleaned


def get_job_state(job_id: str) -> str | None:
    """Query the state of a Slurm job.

    Returns the normalized job state string, or None if the job cannot be found
    or Slurm commands are unavailable.
    """
    # First try sacct for completed jobs
    try:
        result = run_command(["sacct", "-j", job_id, "--format=State", "--noheader"], check=False)
        if result.stdout.strip():
            return normalize_state(result.stdout.strip().splitlines()[0])
    except FileNotFoundError:
        # sacct command not found - continue to fallback
        pass

    # Fallback to squeue for running jobs
    try:
        result = run_command(["squeue", "-j", job_id, "-h", "-o", "%T"], check=False)
        if result.stdout.strip():
            return normalize_state(result.stdout.strip().splitlines()[0])
    except FileNotFoundError:
        # squeue command not found - Slurm may not be installed
        pass

    # Job not found in either command, or Slurm commands unavailable
    return None


def wait_for_success(job_id: str) -> None:
    terminal_states = {
        "COMPLETED": True,
        "FAILED": False,
        "CANCELLED": False,
        "TIMEOUT": False,
        "OUT_OF_MEMORY": False,
        "NODE_FAIL": False,
    }

    while True:
        state = get_job_state(job_id)
        if state is None:
            time.sleep(JOB_POLL_SECONDS)
            continue

        if state in terminal_states:
            if terminal_states[state]:
                return
            raise RuntimeError(f"Job {job_id} finished unsuccessfully with state: {state}")

        time.sleep(JOB_POLL_SECONDS)


def copy_ser_data(experiment, comm_size: int, job_id: str | None = None) -> Path:
    exp_dir = get_f90exp_dir(experiment)
    src_dir = exp_dir / "ser_data"
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing ser_data folder: {src_dir}")

    # Flattened structure: OUTPUT_ROOT/mpitaskX_expname_vYY/
    dest_dir = get_serdata_dst_dir(experiment, comm_size)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    # Copy ser_data folder
    shutil.copytree(src_dir, dest_dir / definitions.SERIALIZED_DATA_SUBDIR)

    # Copy NAMELIST files
    namelist_files = sorted(exp_dir.glob("NAMELIST_*"))
    for src_file in namelist_files:
        if src_file.is_file():
            shutil.copy2(src_file, dest_dir / src_file.name)

    # Copy LOG file if available
    if job_id is not None:
        log_file = RUNSCRIPTS_DIR / f"LOG.{get_slurmscript_name(experiment)}.{job_id}.o"
        if log_file.is_file():
            shutil.copy2(log_file, dest_dir / log_file.name)

    return dest_dir


def tar_folder(folder: Path, experiment: definitions.Experiment, comm_size: int) -> Path:
    tar_path = get_tar_path(experiment, comm_size)

    with tarfile.open(tar_path, "w:gz") as tar:
        # Add only the contents of the folder (NAMELIST files and ser_data), not the folder itself
        for item in folder.iterdir():
            tar.add(item, arcname=item.name)

    return tar_path


def generate_update_script(experiment: definitions.Experiment) -> None:
    # copy namelist file from repo to build_dir
    shutil.copy2(
        ICONF90_DIR / "run" / get_nmlfile_name(experiment),
        RUNSCRIPTS_DIR / get_nmlfile_name(experiment),
    )

    # run make_runscript
    cmd = ["./make_runscripts", get_f90exp_name(experiment)]
    _ = run_command(cmd, cwd=BUILD_DIR)


def run_experiment(experiment: definitions.Experiment, comm_size: int) -> None:
    """Execute a single experiment with the given communicator size."""
    try:
        # Clean up previous experiment output
        cleanup_exp_output(experiment, comm_size)

        generate_update_script(experiment)

        script_path = RUNSCRIPTS_DIR / get_slurmscript_name(experiment)
        if not script_path.exists():
            raise FileNotFoundError(f"Missing slurm script: {script_path}")

        # Parse extra MPI ranks from the script
        extra_mpi_ranks = parse_extra_mpi_ranks(script_path, comm_size)

        log_status(
            f"Setting up {experiment.name} with {comm_size} ranks"
            + (f" + {extra_mpi_ranks} extra" if extra_mpi_ranks > 0 else "")
        )
        update_slurm_variables(script_path)
        update_slurm_ranks(script_path, comm_size, extra_mpi_ranks)

        log_status(f"Submitting {experiment.name} with {comm_size} ranks")
        job_id = submit_job(script_path)

        log_status(f"Waiting for {experiment.name} (ranks={comm_size}, job_id={job_id})")
        wait_for_success(job_id)

        log_status(f"Copying ser_data for {experiment.name} with {comm_size} ranks")
        dest_dir = copy_ser_data(experiment, comm_size, job_id)

        log_status(f"Creating tar archive for {experiment.name} with {comm_size} ranks")
        tar_folder(dest_dir, experiment, comm_size)

        log_status(f"Completed {experiment.name} with {comm_size} ranks")
    except Exception as e:
        log_status(f"ERROR in {experiment.name} with {comm_size} ranks: {e}")
        raise


@cli.command()
def run_experiment_series() -> None:
    """Run the serialization experiment series."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    total_tasks = len(EXPERIMENTS) * len(COMM_SIZES)
    log_status(
        f"Starting experiment series with {total_tasks} tasks ({len(EXPERIMENTS)} experiments x {len(COMM_SIZES)} communicator sizes)"
    )

    for rank_idx, comm_size in enumerate(COMM_SIZES, 1):
        num_experiments = len(EXPERIMENTS)
        log_status(
            f"Starting communicator size {rank_idx}/{len(COMM_SIZES)}: {comm_size} ranks ({num_experiments} experiments parallel)"
        )

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = []

            for experiment in EXPERIMENTS:
                future = executor.submit(run_experiment, experiment, comm_size)
                futures.append(future)

            log_status(
                f"All {len(futures)} experiments queued for {comm_size} ranks, waiting for completion..."
            )

            # Wait for all futures to complete and collect exceptions
            for future in futures:
                future.result()  # Re-raises any exceptions from the thread

        log_status(f"Completed communicator size {rank_idx}/{len(COMM_SIZES)}: {comm_size} ranks")

    log_status(f"All {total_tasks} tasks completed successfully!")


if __name__ == "__main__":
    cli()
