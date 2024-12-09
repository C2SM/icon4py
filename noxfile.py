# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import nox

nox.options.default_venv_backend = "uv"
# nox.options.sessions = ["lint", "test"]


def install_session_venv(
    session: nox.Session,
    *args: str | Sequence[str],
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
) -> None:
    """Install session packages using uv."""
    session.run_install(
        "uv",
        "sync",
        "--no-dev",
        *(f"--extra={e}" for e in extras),
        *(f"--group={g}" for g in groups),
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    for item in args:
        session.run_install(
            "uv",
            "pip",
            "install",
            *((item,) if isinstance(item, str) else item),
            env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
        )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("selection", ["regular_tests", "slow_tests"])
def test_common(session: nox.Session, selection: str) -> None:
    """Run tests for the common package of the icon4py model."""
    install_session_venv(session, extras=["all"], groups=["test"])

    pytest_args = []
    match selection:
        case "regular_tests":
            pytest_args += ["-m", "not slow_tests"]
        case "slow_tests":
            pytest_args += ["-m", "slow_tests"]

    with session.chdir("model/common"):
        session.run(
            "pytest",
            "-sv",
            "--benchmark-skip",
            "-n",
            session.env.get("NUM_PROCESSES", "auto"),
            *pytest_args,
            *session.posargs,
        )

@nox.session(python=["3.10", "3.11"])
def test_driver(session: nox.Session) -> None:
    """Run tests for the driver."""
    install_session_venv(session, extras=["all"], groups=["test"])

    with session.chdir("model/driver"):
        session.run(
            "pytest",
            "-sv",
            "-n",
            session.env.get("NUM_PROCESSES", "auto"),
            *session.posargs,
        )

@nox.session(python=["3.10", "3.11"])
def test_tools(session: nox.Session) -> None:
    """Run tests for the Fortran integration tools."""
    install_session_venv(session, extras=["all"], groups=["test"])

    with session.chdir("tools"):
        session.run(
            "pytest",
            "-sv",
            "-n",
            session.env.get("NUM_PROCESSES", "auto"),
            *session.posargs,
        )



# [testenv:run_stencil_tests]
# commands =
#     pytest -v -m "not slow_tests" --cov --cov-append atmosphere/diffusion/tests/diffusion_stencil_tests --benchmark-skip -n {env:NUM_PROCESSES:1} {posargs}
#     pytest -v -m "not slow_tests" --cov --cov-append atmosphere/dycore/tests/dycore_stencil_tests --benchmark-skip -n {env:NUM_PROCESSES:1} {posargs}
#     pytest -v -m "not slow_tests" --cov --cov-append atmosphere/advection/tests/advection_stencil_tests --benchmark-skip -n {env:NUM_PROCESSES:1} {posargs}

# [testenv:run_benchmarks]
# commands =
#     pytest -v -m "not slow_tests" atmosphere/diffusion/tests/diffusion_stencil_tests --benchmark-only {posargs}
#     pytest -v -m "not slow_tests" atmosphere/dycore/tests/dycore_stencil_tests --benchmark-only {posargs}
#     pytest -v -m "not slow_tests" atmosphere/advection/tests/advection_stencil_tests --benchmark-only {posargs}

# [testenv:run_model_tests]
# commands =
#     pytest -v -m "not slow_tests" --datatest {posargs}

# addopts = ["-p", "icon4py.model.testing.pytest_config"]
