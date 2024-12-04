# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "test"]


def session_install(
    session: nox.Session, *, groups: tuple[str, ...]= (), requirements: tuple[tuple[str, ...], ...] = ()
) -> None:
    """Install session packages using uv."""
    session.run_install(
        "uv",
        "sync",
        "--no-dev",
        *(f"--group={g}" for g in groups),
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    for item in requirements:
        session.run_install(
            "uv",
            "pip",
            "install",
            *item
        )


@nox.session(python=["3.10", "3.11"])
def test_tools(session: nox.Session) -> None:
    """Run the unit and regular tests for the integration tools."""
    session_install(session, groups=("test",))
    session.run("pytest", "-sv", "-n", session.env.get("NUM_PROCESSES", "1"), *session.posargs)


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("package", ["atmosphere.dycore", "atmosphere.advection", "atmosphere.diffusion"])
def tests_model(session: nox.Session, package: str) -> None:
    """Run the unit and regular tests for the model."""
    session_install(session, groups=("test",))
    session.run(
        "pytest", "-v", "-m", "not slow_tests", "--cov", "--cov-append", "--benchmark-skip", "-n",  session.env.get("NUM_PROCESSES", "1"),
        "atmosphere/diffusion/tests/diffusion_stencil_tests", *session.posargs
    )
    pytest -v -m "not slow_tests" --cov --cov-append atmosphere/dcore/tests/dycore_stencil_tests --benchmark-skip -n {env:NUM_PROCESSES:1} {posargs}
    pytest -v -m "not slow_tests" --cov --cov-append atmosphere/advection/tests/advection_stencil_tests --benchmark-skip -n {env:NUM_PROCESSES:1} {posargs}





@nox.session(python=["3.10", "3.11"])
@nox.parametrize("package", ["atmosphere.dycore", "atmosphere.advection", "atmosphere.diffusion"])
def tests_model(session: nox.Session, package: str) -> None:
    """Run the unit and regular tests for the model."""
    session_install(session, groups=("test",))
    session.run(
        "pytest", "-v", "-m", "not slow_tests", "--cov", "--cov-append", "--benchmark-skip", "-n",  session.env.get("NUM_PROCESSES", "1"),
        "atmosphere/diffusion/tests/diffusion_stencil_tests", *session.posargs
    )
    pytest -v -m "not slow_tests" --cov --cov-append atmosphere/dcore/tests/dycore_stencil_tests --benchmark-skip -n {env:NUM_PROCESSES:1} {posargs}
    pytest -v -m "not slow_tests" --cov --cov-append atmosphere/advection/tests/advection_stencil_tests --benchmark-skip -n {env:NUM_PROCESSES:1} {posargs}



[testenv:run_stencil_tests]
commands =
    pytest -v -m "not slow_tests" --cov --cov-append atmosphere/diffusion/tests/diffusion_stencil_tests --benchmark-skip -n {env:NUM_PROCESSES:1} {posargs}
    pytest -v -m "not slow_tests" --cov --cov-append atmosphere/dycore/tests/dycore_stencil_tests --benchmark-skip -n {env:NUM_PROCESSES:1} {posargs}
    pytest -v -m "not slow_tests" --cov --cov-append atmosphere/advection/tests/advection_stencil_tests --benchmark-skip -n {env:NUM_PROCESSES:1} {posargs}

[testenv:run_benchmarks]
commands =
    pytest -v -m "not slow_tests" atmosphere/diffusion/tests/diffusion_stencil_tests --benchmark-only {posargs}
    pytest -v -m "not slow_tests" atmosphere/dycore/tests/dycore_stencil_tests --benchmark-only {posargs}
    pytest -v -m "not slow_tests" atmosphere/advection/tests/advection_stencil_tests --benchmark-only {posargs}

[testenv:run_model_tests]
commands =
    pytest -v -m "not slow_tests" --datatest {posargs}


addopts = ["-p", "icon4py.model.testing.pytest_config"]
