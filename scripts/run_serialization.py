#!/usr/bin/env python3
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

from icon4py.model.testing.definitions import Experiment, Experiments


# ======================================
# USER CONFIGURATION
# ======================================
MPI_RANKS: list[int] = [1, 2, 4]

EXPERIMENTS = [
	Experiments.MCH_CH_R04B09,
	Experiments.JW,
	Experiments.EXCLAIM_APE,
	Experiments.GAUSS3D,
	Experiments.WEISMAN_KLEMP_TORUS,
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
OUTPUT_ROOT = EXPERIMENTS_DIR / "serialized_runs"

# Maximum concurrent threads for running experiments
MAX_THREADS: int = 5

# ======================================
# END USER CONFIGURATION
# ======================================


# Serialization helper functions
def get_slurm_name(exp: Experiment) -> str:
	return f"{exp.name}_sb"


def get_script_name(exp: Experiment) -> str:
	return f"exp.{get_slurm_name(exp)}.run"


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
	return subprocess.run(cmd, check=check, text=True, capture_output=True)


def log_status(message: str) -> None:
	"""Log a status message with timestamp."""
	timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	print(f"[{timestamp}] {message}")


def update_slurm_variables(script_path: Path) -> None:
	"""Update SBATCH directives in the Slurm script (partition, account, time, uenv, view)."""
	original = script_path.read_text()
	updated = original
	
	# Find the position after #SBATCH --job-name= line
	job_name_match = re.search(r"^#SBATCH\s+--job-name=.*$", updated, flags=re.MULTILINE)
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
	updated = re.sub(r"^#SBATCH\s+--partition=.*$\n?", "", updated, flags=re.MULTILINE)
	updated = re.sub(r"^#SBATCH\s+--account=.*$\n?", "", updated, flags=re.MULTILINE)
	updated = re.sub(r"^#SBATCH\s+--time=.*$\n?", "", updated, flags=re.MULTILINE)
	updated = re.sub(r"^#SBATCH\s+--uenv=.*$\n?", "", updated, flags=re.MULTILINE)
	updated = re.sub(r"^#SBATCH\s+--view=.*$\n?", "", updated, flags=re.MULTILINE)
	
	# Re-find job-name position in the cleaned text
	job_name_match = re.search(r"^(#SBATCH\s+--job-name=.*$)", updated, flags=re.MULTILINE)
	if not job_name_match:
		raise RuntimeError("Could not find #SBATCH --job-name= line in script")
	
	# Insert new lines after the job-name line
	insertion_point = job_name_match.end()
	updated = updated[:insertion_point] + "\n" + new_lines + updated[insertion_point:]
	
	script_path.write_text(updated)


def update_slurm_ranks(script_path: Path, mpi_ranks: int, reserved_ranks: int = 0) -> None:
	"""Update ranks in the Slurm script (ntasks-per-node and mpi_procs_pernode).
	
	Args:
		script_path: Path to the Slurm script
		mpi_ranks: Base number of MPI ranks
		reserved_ranks: Additional ranks reserved for special operations (e.g., pre-fetch)
	"""
	total_ranks = mpi_ranks + reserved_ranks
	original = script_path.read_text()

	updated = original
	
	# Update #SBATCH --ntasks-per-node=X
	updated = re.sub(
		r"^#SBATCH\s+--ntasks-per-node\s*=\s*\d+\s*$",
		f"#SBATCH --ntasks-per-node={total_ranks}",
		updated,
		flags=re.MULTILINE,
	)
	
	# Update : ${no_of_nodes:=1} ${mpi_procs_pernode:=X}
	updated = re.sub(
		r"^:\s+\$\{no_of_nodes:=\d+\}\s+\$\{mpi_procs_pernode:=\d+\}\s*$",
		f": ${{no_of_nodes:=1}} ${{mpi_procs_pernode:={total_ranks}}}",
		updated,
		flags=re.MULTILINE,
	)

	script_path.write_text(updated)


def submit_job(script_path: Path) -> str:
	cmd = ["sbatch", str(script_path)]
	result = run_command(cmd)
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
	# First try sacct for completed jobs
	try:
		result = run_command(["sacct", "-j", job_id, "--format=State", "--noheader"], check=False)
		if result.stdout.strip():
			return normalize_state(result.stdout.strip().splitlines()[0])
	except FileNotFoundError:
		pass

	# Fallback to squeue for running jobs
	try:
		result = run_command(["squeue", "-j", job_id, "-h", "-o", "%T"], check=False)
		if result.stdout.strip():
			return normalize_state(result.stdout.strip().splitlines()[0])
	except FileNotFoundError:
		pass

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


def copy_ser_data(exp, mpi_ranks: int) -> Path:
	exp_dir = EXPERIMENTS_DIR / get_slurm_name(exp)
	src_dir = exp_dir / "ser_data"
	if not src_dir.exists():
		raise FileNotFoundError(f"Missing ser_data folder: {src_dir}")

	dest_dir = OUTPUT_ROOT / f"mpirank{mpi_ranks}" / exp.name_with_version
	dest_dir.parent.mkdir(parents=True, exist_ok=True)

	if dest_dir.exists():
		shutil.rmtree(dest_dir)

	dest_dir.mkdir(parents=True, exist_ok=True)
	shutil.copytree(src_dir, dest_dir / "ser_data")
	
	# Copy NAMELIST files
	namelist_files = sorted(exp_dir.glob("NAMELIST_*"))
	for src_file in namelist_files:
		if src_file.is_file():
			shutil.copy2(src_file, dest_dir / src_file.name)
	
	return dest_dir


def tar_folder(folder: Path, exp) -> Path:
	tar_path = folder.parent / exp.archive_filename
	if tar_path.exists():
		tar_path.unlink()

	with tarfile.open(tar_path, "w:gz") as tar:
		tar.add(folder, arcname=folder.name)

	return tar_path


def run_experiment(exp: Experiment, mpi_ranks: int) -> None:
	"""Execute a single experiment with the given rank configuration."""
	try:
		script_path = RUNSCRIPTS_DIR / get_script_name(exp)
		if not script_path.exists():
			raise FileNotFoundError(f"Missing slurm script: {script_path}")

		log_status(
			f"Setting up {exp.name} with {mpi_ranks} ranks"
			+ (f" + {exp.reserved_ranks} reserved" if exp.reserved_ranks > 0 else "")
		)
		update_slurm_variables(script_path)
		update_slurm_ranks(script_path, mpi_ranks, exp.reserved_ranks)
		
		log_status(f"Submitting {exp.name} with {mpi_ranks} ranks")
		job_id = submit_job(script_path)
		
		log_status(f"Waiting for {exp.name} (ranks={mpi_ranks}, job_id={job_id})")
		wait_for_success(job_id)
		
		log_status(f"Copying ser_data for {exp.name} with {mpi_ranks} ranks")
		dest_dir = copy_ser_data(exp, mpi_ranks)
		
		log_status(f"Creating tar archive for {exp.name} with {mpi_ranks} ranks")
		tar_folder(dest_dir, exp)
		
		log_status(f"Completed {exp.name} with {mpi_ranks} ranks")
	except Exception as e:
		log_status(f"ERROR in {exp.name} with {mpi_ranks} ranks: {e}")
		raise


def run_experiment_series() -> None:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	os.chdir(RUNSCRIPTS_DIR)
	
	total_tasks = len(EXPERIMENTS) * len(MPI_RANKS)
	log_status(f"Starting experiment series with {total_tasks} tasks ({len(EXPERIMENTS)} experiments × {len(MPI_RANKS)} rank configs)")
	
	for rank_idx, mpi_ranks in enumerate(MPI_RANKS, 1):
		num_exps = len(EXPERIMENTS)
		log_status(
			f"Starting rank config {rank_idx}/{len(MPI_RANKS)}: {mpi_ranks} ranks ({num_exps} experiments parallel)"
		)
		
		with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
			futures = []
			
			for exp in EXPERIMENTS:
				future = executor.submit(run_experiment, exp, mpi_ranks)
				futures.append(future)
			
			log_status(f"All {len(futures)} experiments queued for {mpi_ranks} ranks, waiting for completion...")
			
			# Wait for all futures to complete and collect exceptions
			for future in futures:
				future.result()  # Re-raises any exceptions from the thread
		
		log_status(f"Completed rank config {rank_idx}/{len(MPI_RANKS)}: {mpi_ranks} ranks")
	
	log_status(f"All {total_tasks} tasks completed successfully!")


if __name__ == "__main__":
	run_experiment_series()
