# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import datetime

import pytest
from gt4py import next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.component import MuphysComponent
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import SPECIES
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import common, run_full_muphys
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.testing import test_utils
from icon4py.model.testing.fixtures.datatest import backend_like

from . import utils
from .utils import download_test_data


_T0 = datetime.datetime(2024, 1, 1, 0, 0, 0)
_MINI = utils.MuphysExperiment(name="mini", type=utils.ExperimentType.FULL_MUPHYS)


@pytest.mark.uses_concat_where
@pytest.mark.datatest
@pytest.mark.level("integration")
@pytest.mark.parametrize("experiment", [_MINI], ids=lambda e: e.name)
def test_granule_matches_direct_muphys(
    backend_like: model_backends.BackendLike,
    experiment: utils.MuphysExperiment,
) -> None:
    allocator = model_backends.get_allocator(backend_like)
    inp = common.GraupelInput.load(filename=experiment.input_file, allocator=allocator)

    te0 = inp.t.asnumpy().copy()
    q0 = {s: getattr(inp, f"q{s}").asnumpy().copy() for s in SPECIES}

    muphys_program = run_full_muphys.setup_muphys(
        inp=inp,
        dt=experiment.dt,
        qnc=experiment.qnc,
        backend=backend_like,
        single_program=False,
    )

    granule = MuphysComponent(
        ncells=inp.ncells,
        nlev=inp.nlev,
        dtime=datetime.timedelta(seconds=experiment.dt),
        qnc=experiment.qnc,
        backend=backend_like,
        step=muphys_program,
    )
    state = {
        "dz": inp.dz,
        "te": inp.t,
        "p": inp.p,
        "rho": inp.rho,
        "qv": inp.qv,
        "qc": inp.qc,
        "qr": inp.qr,
        "qs": inp.qs,
        "qi": inp.qi,
        "qg": inp.qg,
    }
    out = granule(state, _T0)

    direct = common.GraupelOutput.allocate(
        allocator=allocator,
        domain=gtx.domain({dims.CellDim: inp.ncells, dims.KDim: inp.nlev}),
    )
    muphys_program(
        dz=inp.dz,
        te=inp.t,
        p=inp.p,
        rho=inp.rho,
        q_in=inp.q,
        t_out=direct.t,
        q_out=direct.q,
        pflx=direct.pflx,
        pr=direct.pr,
        ps=direct.ps,
        pi=direct.pi,
        pg=direct.pg,
        pre=direct.pre,
    )

    dt = experiment.dt

    # Reconstructing the updated state as ``old + tendency*dt`` is not bit-exact
    assert test_utils.dallclose(
        te0 + out["tend_temperature"].asnumpy() * dt, direct.t.asnumpy(), atol=1e-15
    )
    for s in SPECIES:
        applied = q0[s] + out[f"tend_q{s}"].asnumpy() * dt
        assert test_utils.dallclose(applied, getattr(direct, f"q{s}").asnumpy(), atol=1e-15)

    assert test_utils.dallclose(out["pflx"].asnumpy(), direct.pflx.asnumpy(), rtol=0.0, atol=0.0)
    for name in ("pr", "ps", "pi", "pg", "pre"):
        assert test_utils.dallclose(
            out[name].asnumpy()[:, -1],
            getattr(direct, name).asnumpy()[:, -1],
            rtol=0.0,
            atol=0.0,
        )
