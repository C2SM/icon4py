# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import pathlib
from typing import Final

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import run_graupel_only
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.testing import data_handling, definitions as testing_defs
from icon4py.model.testing.fixtures.datatest import backend_like


def _path_to_experiment_testdata(experiment: MuphysGraupelExperiment) -> pathlib.Path:
    return testing_defs.get_test_data_root_path() / "muphys_graupel_data" / experiment.name


@dataclasses.dataclass(frozen=True)
class MuphysGraupelExperiment:
    name: str
    uri: str
    dtype: np.dtype
    dt: float = 30.0
    qnc: float = 100.0

    @property
    def input_file(self) -> pathlib.Path:
        return _path_to_experiment_testdata(self) / "input.nc"

    @property
    def reference_file(self) -> pathlib.Path:
        return _path_to_experiment_testdata(self) / "reference.nc"

    def __str__(self):
        return self.name


class Experiments:
    # TODO currently on havogt's polybox
    MINI: Final = MuphysGraupelExperiment(
        name="mini",
        uri="https://polybox.ethz.ch/index.php/s/55oHBDxS2SiqAGN/download/mini.tar.gz",
        dtype=np.float32,
    )
    TINY: Final = MuphysGraupelExperiment(
        name="tiny",
        uri="https://polybox.ethz.ch/index.php/s/5Ceop3iaWkbc7gf/download/tiny.tar.gz",
        dtype=np.float64,
    )
    R2B05: Final = MuphysGraupelExperiment(
        name="R2B05",
        uri="https://polybox.ethz.ch/index.php/s/RBib8rFSEd7Eomo/download/R2B05.tar.gz",
        dtype=np.float32,
    )


@pytest.fixture(autouse=True)
def download_test_data(experiment: MuphysGraupelExperiment) -> None:
    """Downloads test data for an experiment (implicit fixture)."""
    data_handling.download_test_data(_path_to_experiment_testdata(experiment), uri=experiment.uri)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        Experiments.MINI,
        Experiments.TINY,
        Experiments.R2B05,
    ],
    ids=lambda exp: exp.name,
)
def test_graupel_only(
    backend_like: model_backends.BackendLike,
    experiment: MuphysGraupelExperiment,
) -> None:
    inp = run_graupel_only.GraupelInput.load(
        filename=experiment.input_file, allocator=model_backends.get_allocator(backend_like)
    )

    graupel_run_program = run_graupel_only.setup_graupel(
        inp, dt=experiment.dt, qnc=experiment.qnc, backend=backend_like
    )

    out = run_graupel_only.GraupelOutput.allocate(
        allocator=model_backends.get_allocator(backend_like),
        domain=gtx.domain({dims.CellDim: inp.ncells, dims.KDim: inp.nlev}),
    )

    graupel_run_program(
        dz=inp.dz,
        te=inp.t,
        p=inp.p,
        rho=inp.rho,
        qve=inp.qv,
        qce=inp.qc,
        qre=inp.qr,
        qse=inp.qs,
        qie=inp.qi,
        qge=inp.qg,
        t_out=out.t,
        qv_out=out.qv,
        qc_out=out.qc,
        qr_out=out.qr,
        qs_out=out.qs,
        qi_out=out.qi,
        qg_out=out.qg,
        pflx=out.pflx,
        pr=out.pr,
        ps=out.ps,
        pi=out.pi,
        pg=out.pg,
        pre=out.pre,
    )

    ref = run_graupel_only.GraupelOutput.load(
        filename=experiment.reference_file,
        allocator=model_backends.get_allocator(backend_like),
    )

    # TODO check tolerances
    rtol = 1e-14 if experiment.dtype == np.float64 else 1e-7
    atol = 1e-16 if experiment.dtype == np.float64 else 1e-8
    # TODO we run the float32 input experiments with float64

    np.testing.assert_allclose(ref.qv.asnumpy(), out.qv.asnumpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref.qc.asnumpy(), out.qc.asnumpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref.qi.asnumpy(), out.qi.asnumpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref.qr.asnumpy(), out.qr.asnumpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref.qs.asnumpy(), out.qs.asnumpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref.qg.asnumpy(), out.qg.asnumpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref.t.asnumpy(), out.t.asnumpy(), atol=atol, rtol=rtol)
