# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Final

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import common, run_full_muphys
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.testing.fixtures.datatest import backend_like

from . import utils
from .utils import download_test_data


class Experiments:
    # TODO(havogt): the following references need to be checked (and moved to the shared directory),
    # currently they are not verifying
    # https://polybox.ethz.ch/index.php/s/5oNtcQFDcCaNxHH/download/r2b04.tar.gz
    # https://polybox.ethz.ch/index.php/s/mBeAWAQQHSKTkF7/download/r2b04_maxfrac.tar.gz
    # https://polybox.ethz.ch/index.php/s/mBrpE3iBoeek5wc/download/r2b05.tar.gz
    MINI: Final = utils.MuphysExperiment(
        name="mini",
        type=utils.ExperimentType.FULL_MUPHYS,
        uri="https://polybox.ethz.ch/index.php/s/F8bK2C8tkpf8Xy2/download?files=mini.tar.gz",
    )
    # Note: don't use the 'tiny' experiment from graupel_only,
    # as it is not sensitive to saturation adjustment
    # TODO(havogt): double-check that all other experiments actually are sensitive,
    # i.e. reference of full_muphys and graupel_only differ significantly.


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        Experiments.MINI,
        # TODO(havogt): references need to be checked, currently they are not verifying
        # Experiments.R2B04,
        # Experiments.R2B04_MAXFRAC,
        # Experiments.R2B05,
    ],
    ids=lambda exp: exp.name,
)
@pytest.mark.parametrize("single_program", [True, False], ids=lambda sp: f"single_program={sp}")
def test_full_muphys(
    backend_like: model_backends.BackendLike,
    experiment: utils.MuphysExperiment,
    single_program: bool,
) -> None:
    assert experiment.type == utils.ExperimentType.FULL_MUPHYS

    if single_program:
        pytest.xfail("Single program version currently fails verification. Needs investigation.")

    inp = common.GraupelInput.load(
        filename=experiment.input_file, allocator=model_backends.get_allocator(backend_like)
    )

    muphys_program = run_full_muphys.setup_muphys(
        inp,
        dt=experiment.dt,
        qnc=experiment.qnc,
        backend=backend_like,
        single_program=single_program,
    )

    out = common.GraupelOutput.allocate(
        allocator=model_backends.get_allocator(backend_like),
        domain=gtx.domain({dims.CellDim: inp.ncells, dims.KDim: inp.nlev}),
    )

    muphys_program(
        dz=inp.dz,
        te=inp.t,
        p=inp.p,
        rho=inp.rho,
        q_in=inp.q,
        t_out=out.t,
        q_out=out.q,
        pflx=out.pflx,
        pr=out.pr,
        ps=out.ps,
        pi=out.pi,
        pg=out.pg,
        pre=out.pre,
    )

    ref = common.GraupelOutput.load(
        filename=experiment.reference_file, allocator=model_backends.get_allocator(backend_like)
    )

    rtol = 1e-14
    atol = 1e-16

    np.testing.assert_allclose(ref.qv.asnumpy(), out.qv.asnumpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref.qc.asnumpy(), out.qc.asnumpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref.qi.asnumpy(), out.qi.asnumpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref.qr.asnumpy(), out.qr.asnumpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref.qs.asnumpy(), out.qs.asnumpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref.qg.asnumpy(), out.qg.asnumpy(), atol=atol, rtol=rtol)
    np.testing.assert_allclose(ref.t.asnumpy(), out.t.asnumpy(), atol=atol, rtol=rtol)
