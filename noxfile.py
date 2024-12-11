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


@nox.session(python=["3.10", "3.11"])
@nox.parametrize(
    "subpackage",
    ["advection", "diffusion", "dycore", "subgrid_scale_physics/microphysics"],
)
def benchmark_atmosphere(session: nox.Session, subpackage: str) -> None:
    """Run pytest benchmarks for the `model.atmosphere` subpackages."""
    _install_session_venv(session, extras=["all"], groups=["test"])

    with session.chdir(f"model/atmosphere/{subpackage}"):
        session.run("pytest", "-sv", "--benchmark-only", *session.posargs)


@nox.session(python=["3.10", "3.11"])
@nox.parametrize(
    "subpackage",
    ["advection", "diffusion", "dycore", "subgrid_scale_physics/microphysics"],
)
@nox.parametrize("selection", ["regular_tests", "slow_tests"])
def test_atmosphere(session: nox.Session, subpackage: str, selection: str) -> None:
    """Run tests for the `model.atmosphere` subpackages."""
    _install_session_venv(session, extras=["all"], groups=["test"])

    assert (
        selection in _SELECTION_PARAM_TO_PYTEST_MARKERS
    ), f"Invalid test selection argument: {selection}"
    pytest_args = sum(
        (["-m", tag] for tag in _SELECTION_PARAM_TO_PYTEST_MARKERS[selection]), start=[]
    )

    with session.chdir(f"model/atmosphere/{subpackage}"):
        session.run(
            "pytest",
            "-sv",
            "-n",
            session.env.get("NUM_PROCESSES", "auto"),
            "--benchmark-skip",
            *pytest_args,
            *session.posargs,
        )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("selection", ["regular_tests", "slow_tests"])
def test_atmosphere_advection(session: nox.Session, selection: str) -> None:
    session.notify(
        f"test_atmosphere-{session.python}(selection='{selection}', subpackage='advection')"
    )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("selection", ["regular_tests", "slow_tests"])
def test_atmosphere_diffusion(session: nox.Session, selection: str) -> None:
    session.notify(
        f"test_atmosphere-{session.python}(selection='{selection}', subpackage='diffusion')"
    )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("selection", ["regular_tests", "slow_tests"])
def test_atmosphere_dycore(session: nox.Session, selection: str) -> None:
    session.notify(
        f"test_atmosphere-{session.python}(selection='{selection}', subpackage='dycore')"
    )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("selection", ["regular_tests", "slow_tests"])
def test_atmosphere_microphysics(session: nox.Session, selection: str) -> None:
    session.notify(
        f"test_atmosphere-{session.python}(selection='{selection}', subpackage='subgrid_scale_physics/microphysics')"
    )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("selection", ["regular_tests", "slow_tests"])
def test_common(session: nox.Session, selection: str) -> None:
    """Run tests for the common package of the icon4py model."""
    _install_session_venv(session, extras=["all"], groups=["test"])

    assert (
        selection in _SELECTION_PARAM_TO_PYTEST_MARKERS
    ), f"Invalid test selection argument: {selection}"
    pytest_args = sum(
        (["-m", tag] for tag in _SELECTION_PARAM_TO_PYTEST_MARKERS[selection]), start=[]
    )

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
    _install_session_venv(session, extras=["all"], groups=["test"])

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
    _install_session_venv(session, extras=["all"], groups=["test"])

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


_SELECTION_PARAM_TO_PYTEST_MARKERS: dict[str, list[str]] = {
    "regular_tests": ["not slow_tests"],
    "slow_tests": ["slow_tests"],
}


def _install_session_venv(
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
