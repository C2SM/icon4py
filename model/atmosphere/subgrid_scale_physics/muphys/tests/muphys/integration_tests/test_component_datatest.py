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


class _FullDomainGrid:
    """Grid stand-in for the grid-less muphys netCDF data: prognostic bounds = full domain."""

    def __init__(self, num_cells: int, num_levels: int) -> None:
        self.num_cells = num_cells
        self.num_levels = num_levels

    def start_index(self, domain: object) -> gtx.int32:
        return gtx.int32(0)

    def end_index(self, domain: object) -> gtx.int32:
        return gtx.int32(self.num_cells)


@pytest.mark.uses_concat_where
@pytest.mark.datatest
@pytest.mark.level("integration")
@pytest.mark.parametrize("experiment", [_MINI], ids=lambda e: e.name)
def test_granule_matches_direct_muphys(
    backend_like: model_backends.BackendLike,
    experiment: utils.MuphysExperiment,
) -> None:
    allocator = model_backends.get_allocator(backend_like)
    graupel_input = common.GraupelInput.load(filename=experiment.input_file, allocator=allocator)

    te0 = graupel_input.t.asnumpy().copy()
    q0 = {s: getattr(graupel_input, f"q{s}").asnumpy().copy() for s in SPECIES}

    muphys_program = run_full_muphys.setup_muphys(
        inp=graupel_input,
        dt=experiment.dt,
        qnc=experiment.qnc,
        backend=backend_like,
        single_program=False,
    )

    granule = MuphysComponent(
        grid=_FullDomainGrid(graupel_input.ncells, graupel_input.nlev),  # type: ignore[arg-type]  # mini data has no icon grid
        dtime=datetime.timedelta(seconds=experiment.dt),
        qnc=experiment.qnc,
        backend=backend_like,  # type: ignore[arg-type]  # BackendLike includes DeviceType/dict not accepted by MuphysComponent
        step=muphys_program,
    )
    state = {
        "dz": graupel_input.dz,
        "te": graupel_input.t,
        "p": graupel_input.p,
        "rho": graupel_input.rho,
        "qv": graupel_input.qv,
        "qc": graupel_input.qc,
        "qr": graupel_input.qr,
        "qs": graupel_input.qs,
        "qi": graupel_input.qi,
        "qg": graupel_input.qg,
    }
    out = granule(state, _T0)  # type: ignore[arg-type]  # GT4Py Field/DataField Protocol mismatch

    direct = common.GraupelOutput.allocate(
        allocator=allocator,
        domain=gtx.domain({dims.CellDim: graupel_input.ncells, dims.KDim: graupel_input.nlev}),
    )
    direct.t.ndarray[...] = graupel_input.t.ndarray  # type: ignore[index]  # NDArrayObject Protocol lacks item assignment
    for s in SPECIES:
        getattr(direct, f"q{s}").ndarray[...] = getattr(graupel_input, f"q{s}").ndarray
    muphys_program(
        dz=graupel_input.dz,
        te=direct.t,
        p=graupel_input.p,
        rho=graupel_input.rho,
        q_in=direct.q,
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
        te0 + out["tend_temperature"].asnumpy() * dt,  # type: ignore[attr-defined]  # DataField protocol lacks asnumpy; concrete field at runtime
        direct.t.asnumpy(),
        atol=1e-15,
    )
    for s in SPECIES:
        applied = q0[s] + out[f"tend_q{s}"].asnumpy() * dt  # type: ignore[attr-defined]  # DataField protocol lacks asnumpy; concrete field at runtime
        assert test_utils.dallclose(
            applied,
            getattr(direct, f"q{s}").asnumpy(),
            atol=1e-15,
        )

    assert test_utils.dallclose(
        out["pflx"].asnumpy(),  # type: ignore[attr-defined]  # DataField protocol lacks asnumpy; concrete field at runtime
        direct.pflx.asnumpy(),  # type: ignore[union-attr]  # GraupelOutput field may be None per protocol; concrete field at runtime
        rtol=0.0,
        atol=0.0,
    )
    for name in ("pr", "ps", "pi", "pg", "pre"):
        assert test_utils.dallclose(
            out[name].asnumpy()[:, -1],  # type: ignore[attr-defined]  # DataField protocol lacks asnumpy; concrete field at runtime
            getattr(direct, name).asnumpy()[:, -1],
            rtol=0.0,
            atol=0.0,
        )
