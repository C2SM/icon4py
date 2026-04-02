# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Run serialization jobs, collect ser_data and NAMELISTS, and archive outputs."""

from __future__ import annotations

import pathlib
import os

from icon4py.model.testing import datatest_utils as dt_utils, definitions

# ======================================
# USER CONFIGURATION
# ======================================

COMM_SIZES: list[int] = [1, 2, 4]

EXPERIMENTS = [
    definitions.Experiments.MCH_CH_R04B09,
    definitions.Experiments.JW,
    definitions.Experiments.EXCLAIM_APE,
    definitions.Experiments.GAUSS3D,
    definitions.Experiments.WEISMAN_KLEMP_TORUS,
]

# Slurm settings
SBATCH_PARTITION = "normal"
SBATCH_TIME = "00:15:00"
SBATCH_ACCOUNT = "cwd01"
SBATCH_UENV = "icon/25.2:v3"
SBATCH_UENV_VIEW = "default"
JOB_POLL_SECONDS = 10

# Base directories (adjust if needed)
PROJECTS_DIR = pathlib.Path(os.environ.get("SCRATCH", str(pathlib.Path.home() / "projects")))
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
