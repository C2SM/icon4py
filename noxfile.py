# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import json
import glob
import re
from collections.abc import Sequence
from typing import Final, Literal, TypeAlias
from datetime import datetime

import nox

# -- nox configuration --
nox.options.default_venv_backend = "uv"
nox.options.sessions = ["test_model", "test_tools"]


# -- Parameter sets --
ModelSubpackagePath: TypeAlias = Literal[
    # "atmosphere/advection",
    "atmosphere/diffusion",
    "atmosphere/dycore",
    "atmosphere/subgrid_scale_physics/microphysics",
    "common",
    "driver",
    # "testing", #TODO: Add tests to testing subpackage
]
MODEL_SUBPACKAGE_PATHS: Final[Sequence[nox.Param]] = [
    nox.param(arg, id=arg.split("/")[-1]) for arg in ModelSubpackagePath.__args__
]

ModelTestsSubset: TypeAlias = Literal["datatest", "stencils", "basic"]
MODEL_TESTS_SUBSETS: Final[Sequence[str]] = [
        nox.param(arg, id=arg, tags=[arg]) for arg in ModelTestsSubset.__args__
]
# -- nox sessions --
#: This should just be `pytest.ExitCode.NO_TESTS_COLLECTED` but `pytest`
#: is not guaranteed to be available in the venv where `nox` is running.
NO_TESTS_COLLECTED_EXIT_CODE: Final = 5

# Model benchmark sessions
# TODO(egparedes): Add backend parameter
# TODO(edopao,egparedes): Change 'extras' back to 'all' once mpi4py can be compiled with hpc_sdk
@nox.session(python=["3.10", "3.11"])
@nox.parametrize("subpackage", MODEL_SUBPACKAGE_PATHS)
def benchmark_model(session: nox.Session, subpackage: ModelSubpackagePath) -> None:
    """Run pytest benchmarks for selected icon4py model subpackages."""
    _install_session_venv(session, extras=["dace", "io", "testing"], groups=["test"])

    results_json_path = os.path.abspath(f"pytest_benchmark_results_{session.python}_{subpackage.replace('/', '_')}.json")
    with session.chdir(f"model/{subpackage}"):
        session.run(
            *f"pytest \
            -v \
            --benchmark-only \
            --benchmark-warmup=on \
            --benchmark-warmup-iterations=30 \
            --benchmark-json={results_json_path}".split(),
            *session.posargs,
            success_codes=[0, NO_TESTS_COLLECTED_EXIT_CODE],
        )

def merge_pytest_benchmark_results(bencher_json_file_name: str, session_python: str) -> None:
    """Gather all benchmark results files and merge them into a single file."""
    merged_results = {"benchmarks": []}
    for file in glob.glob(f"pytest_benchmark_results_{session_python}_*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                merged_results["benchmarks"].extend(data["benchmarks"])
                # preserve the first file's metadata for a valid pytest-benchmark json file
                for key in data.keys():
                    if key != "benchmarks" and key not in merged_results:
                        merged_results[key] = data[key]
        except:
            # Empty file, i.e. no benchmarks
            continue
    with open(bencher_json_file_name, "w") as f:
        json.dump(merged_results, f, indent=4)

@nox.session(python=["3.10", "3.11"],
             requires=["benchmark_model-{python}" + f"({subpackage.id})" for subpackage in MODEL_SUBPACKAGE_PATHS])
def __bencher_baseline_CI(session: nox.Session) -> None:
    """
    Run pytest benchmarks and upload them using Bencher (https://bencher.dev/) (cloud or self-hosted).
    This session is used only on the main branch to create the historical baseline.
    The historical baseline is used to compare the performance of the code in the PRs.
    Alerts are raised if there is performance regression according to the thresholds.
    Note: This session is intended to be run from the CI only -bencher and suitable env vars are needed-.
    """
    bencher_json_file_name = f"merged_benchmark_results_{session.python}.json"
    merge_pytest_benchmark_results(bencher_json_file_name, session.python)

    session.run(
        *f"bencher run \
        --threshold-measure latency \
        --threshold-test percentage \
        --threshold-max-sample-size 64 \
        --threshold-upper-boundary 0.1 \
        --thresholds-reset \
        --err \
        --file {bencher_json_file_name}".split(),
        env={
            "BENCHER_PROJECT": os.environ["BENCHER_PROJECT"].strip(),  # defined in https://cicd-ext-mw.cscs.ch
            "BENCHER_BRANCH": "main",
            "BENCHER_TESTBED": f"{os.environ['RUNNER']}:{os.environ['SYSTEM_TAG']}:{os.environ['BACKEND']}:{os.environ['GRID']}",
            "BENCHER_ADAPTER": "python_pytest",
            "BENCHER_HOST": os.environ["BENCHER_HOST"].strip(),  # defined in https://cicd-ext-mw.cscs.ch
            "BENCHER_API_TOKEN": os.environ["BENCHER_API_TOKEN"].strip(),
        },
        external=True,
        silent=True,
    )

@nox.session(python=["3.10", "3.11"],
             requires=["benchmark_model-{python}" + f"({subpackage.id})" for subpackage in MODEL_SUBPACKAGE_PATHS])
def __bencher_feature_branch_CI(session: nox.Session) -> None:
    """
    Run pytest benchmarks and upload them using Bencher (https://bencher.dev/) (cloud or self-hosted).
    This session compares the performance of the feature branch with the historical baseline (as built from __bencher_baseline_CI session).
    Alerts are raised if the performance of the feature branch is worse than the historical baseline (according to the thresholds).
    Note: This session is intended to be run from the CI only -bencher and suitable env vars are needed-.
    """
    bencher_json_file_name = f"merged_benchmark_results_{session.python}.json"
    merge_pytest_benchmark_results(bencher_json_file_name, session.python)
    
    bencher_testbed = f"{os.environ['RUNNER']}:{os.environ['SYSTEM_TAG']}:{os.environ['BACKEND']}:{os.environ['GRID']}"
    
    session.run(
        *f"bencher run \
        --start-point main \
        --start-point-clone-thresholds \
        --start-point-reset \
        --err \
        --github-actions {os.environ['GD_COMMENT_TOKEN']} \
        --ci-number {os.environ['PR_ID']} \
        --ci-id run-{bencher_testbed.replace(':', '_')}-{int(datetime.now().strftime('%Y%m%d%H%M%S%f'))} \
        --file {bencher_json_file_name}".split(),
        env={
            "BENCHER_PROJECT": os.environ["BENCHER_PROJECT"].strip(),  # defined in https://cicd-ext-mw.cscs.ch
            "BENCHER_BRANCH": os.environ['FEATURE_BRANCH'].strip(),
            "BENCHER_TESTBED": bencher_testbed,
            "BENCHER_ADAPTER": "python_pytest",
            "BENCHER_HOST": os.environ["BENCHER_HOST"].strip(),  # defined in https://cicd-ext-mw.cscs.ch
            "BENCHER_API_TOKEN": os.environ["BENCHER_API_TOKEN"].strip(),
        },
        external=True,
        silent=True,
    )

# Model test sessions
# TODO(egparedes): Add backend parameter
# TODO(edopao,egparedes): Change 'extras' back to 'all' once mpi4py can be compiled with hpc_sdk
@nox.session(python=["3.10", "3.11"])
@nox.parametrize("subpackage", MODEL_SUBPACKAGE_PATHS)
@nox.parametrize("selection", MODEL_TESTS_SUBSETS)
def test_model(session: nox.Session, selection: ModelTestsSubset, subpackage: ModelSubpackagePath) -> None:
    """Run tests for selected icon4py model subpackages."""
    _install_session_venv(session, extras=["dace", "fortran", "io", "testing"], groups=["test"])

    pytest_args = _selection_to_pytest_args(selection)
    with session.chdir(f"model/{subpackage}"):
        session.run(
            *f"pytest -sv --benchmark-skip -n {os.environ.get('NUM_PROCESSES', 'auto')}".split(),
            *pytest_args,
            *session.posargs,
            success_codes=[0, NO_TESTS_COLLECTED_EXIT_CODE],
        )

# @nox.session(python=["3.10", "3.11"])
# @nox.parametrize("selection", MODEL_TEST_SELECTION)
# def test_testing(session: nox.Session, selection: ModelTestKind) -> None:
#     session.notify(
#         f"test_model-{session.python}(selection='{selection}', subpackage='testing')"
#     )


# Tools test sessions
# TODO(edopao,egparedes): Change 'extras' back to 'all' once mpi4py can be compiled with hpc_sdk
@nox.session(python=["3.10", "3.11"])
@nox.parametrize("datatest", [
    nox.param(False, id="datatest", tags=["datatest"]),
    nox.param(True, id="unittest",)
])
def test_tools(session: nox.Session, datatest: bool) -> None:
    """Run tests for the Fortran integration tools."""
    _install_session_venv(session, extras=["fortran", "io", "testing"], groups=["test"])

    with session.chdir("tools"):
        session.run(
            *f"pytest -sv --benchmark-skip -n {os.environ.get('NUM_PROCESSES', 'auto')} {'--datatest' if datatest else ''}".split(),
            *session.posargs
        )

# -- utils --
def _install_session_venv(
    session: nox.Session,
    *args: str | Sequence[str],
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
) -> None:
    """Install session packages using uv."""
    #TODO(egparedes): remove this workaround once `backend` parameter is added to sessions
    if (env_extras := os.environ.get("ICON4PY_NOX_UV_CUSTOM_SESSION_EXTRAS", "")):
        extras = [*extras, *re.split(r'\W+', env_extras)]
    env = dict(os.environ.items()) | {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    session.run_install(
        "uv",
        "sync",
        *("--python", session.python),
        "--no-dev",
        *(f"--extra={e}" for e in extras),
        *(f"--group={g}" for g in groups),
        env=env
    )
    for item in args:
        session.run_install(
            "uv",
            "pip",
            "install",
            *((item,) if isinstance(item, str) else item),
            env=env
        )

def _selection_to_pytest_args(selection: ModelTestsSubset) -> list[str]:
    pytest_args = []
    
    match selection:
        case "datatest":
            pytest_args.extend(["-k", "not stencil_test", "--datatest"])
        case "stencils":
            pytest_args.extend(["-k", "stencil_tests"])
        case "basic":
            pytest_args.extend(["-k", "not stencil_tests"])
        case _:
            raise AssertionError(f"Invalid selection: {selection}")
        
    return pytest_args