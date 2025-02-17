# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Final, Literal, TypeAlias

import nox

# -- nox configuration --
nox.options.default_venv_backend = "uv"
nox.options.sessions = ["test_model", "test_tools"]


# -- Parameter sets --
ModelSubpackagePath: TypeAlias = Literal[
    "atmosphere/advection",
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

    with session.chdir(f"model/{subpackage}"):
        session.run(*"pytest -sv --benchmark-only".split(), *session.posargs)


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