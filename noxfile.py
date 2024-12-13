# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["test_atmosphere", "test_common", "test_driver", "test_tools"]


@nox.session(python=["3.10", "3.11"])
@nox.parametrize(
    "subpackage",
    ["advection", "diffusion", "dycore", "subgrid_scale_physics/microphysics"],
)
def benchmark_atmosphere(session: nox.Session, subpackage: str) -> None:
    """Run pytest benchmarks for the `model.atmosphere` subpackages."""
    _install_session_venv(session, extras=["all"], groups=["test"])

    with session.chdir(f"model/atmosphere/{subpackage}"):
        session.run(*"pytest -sv --benchmark-only".split(), *session.posargs)


@nox.session(python=["3.10", "3.11"])
@nox.parametrize(
    "subpackage",
    ["advection", "diffusion", "dycore", "subgrid_scale_physics/microphysics"],
)
@nox.parametrize("datatest", [False, True])
def test_atmosphere(session: nox.Session, subpackage: str, datatest: bool) -> None:
    """Run tests for the `model.atmosphere` subpackages."""
    _install_session_venv(session, extras=["all"], groups=["test"])

    with session.chdir(f"model/atmosphere/{subpackage}"):
        session.run(
            *f"pytest -sv --benchmark-skip -n {session.env.get('NUM_PROCESSES', 'auto')} -m {'' if datatest else 'not'} datatest".split(),
            *session.posargs
        )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("datatest", [False, True])
def test_atmosphere_advection(session: nox.Session, datatest: bool) -> None:
    session.notify(
        f"test_atmosphere-{session.python}(datatest='{datatest}', subpackage='advection')"
    )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("datatest", [False, True])
def test_atmosphere_diffusion(session: nox.Session, datatest: bool) -> None:
    session.notify(
        f"test_atmosphere-{session.python}(datatest='{datatest}', subpackage='diffusion')"
    )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("datatest", [False, True])
def test_atmosphere_dycore(session: nox.Session, datatest: bool) -> None:
    session.notify(
        f"test_atmosphere-{session.python}(datatest='{datatest}', subpackage='dycore')"
    )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("datatest", [False, True])
def test_atmosphere_microphysics(session: nox.Session, datatest: bool) -> None:
    session.notify(
        f"test_atmosphere-{session.python}(datatest='{datatest}', subpackage='subgrid_scale_physics/microphysics')"
    )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("datatest", [False, True])
def test_common(session: nox.Session, datatest: bool) -> None:
    """Run tests for the common package of the icon4py model."""
    _install_session_venv(session, extras=["all"], groups=["test"])

    with session.chdir("model/common"):
        session.run(
            *f"pytest -sv -n {session.env.get('NUM_PROCESSES', 'auto')} -m {'' if datatest else 'not'} datatest".split(),
            *session.posargs
        )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("datatest", [False, True])
def test_driver(session: nox.Session, datatest: bool) -> None:
    """Run tests for the driver."""
    _install_session_venv(session, extras=["all"], groups=["test"])

    with session.chdir("model/driver"):
        session.run(
            *f"pytest -sv -n {session.env.get('NUM_PROCESSES', 'auto')} -m {'' if datatest else 'not'} datatest".split(),
            *session.posargs
        )

@nox.session(python=["3.10", "3.11"])
def test_model_datatest(session: nox.Session) -> None:
    session.run(
        *f"pytest -sv -n {session.env.get('NUM_PROCESSES', 'auto')} -m datatest".split(),
        *session.posargs
    )


@nox.session(python=["3.10", "3.11"])
def test_model_operators(session: nox.Session) -> None:
    session.run(
        *f"pytest -sv -n {session.env.get('NUM_PROCESSES', 'auto')} -k 'stencil_tests'".split(),
        *session.posargs
    )


@nox.session(python=["3.10", "3.11"])
@nox.parametrize("datatest", [False, True])
def test_tools(session: nox.Session, datatest: bool) -> None:
    """Run tests for the Fortran integration tools."""
    _install_session_venv(session, extras=["all"], groups=["test"])

    with session.chdir("tools"):
        session.run(
            *f"pytest -sv -n {session.env.get('NUM_PROCESSES', 'auto')} -m {'' if datatest else 'not'} datatest".split(),
            *session.posargs
        )


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
