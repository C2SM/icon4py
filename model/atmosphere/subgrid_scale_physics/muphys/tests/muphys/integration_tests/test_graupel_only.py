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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import common, run_graupel_only
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.testing.fixtures.datatest import backend_like

from . import utils
from .utils import download_test_data


class Experiments:
    MINI: Final = utils.MuphysExperiment(
        name="mini",
        type=utils.ExperimentType.GRAUPEL_ONLY,
        uri="https://polybox.ethz.ch/index.php/s/7B9MWyKTTBrNQBd/download?files=mini.tar.gz",
    )
    TINY: Final = utils.MuphysExperiment(
        name="tiny",
        type=utils.ExperimentType.GRAUPEL_ONLY,
        uri="https://polybox.ethz.ch/index.php/s/7B9MWyKTTBrNQBd/download?files=tiny.tar.gz",
    )
    R2B05: Final = utils.MuphysExperiment(
        name="R2B05",
        type=utils.ExperimentType.GRAUPEL_ONLY,
        uri="https://polybox.ethz.ch/index.php/s/7B9MWyKTTBrNQBd/download?files=R2B05.tar.gz",
    )


@pytest.mark.uses_concat_where
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
    backend_like: model_backends.BackendLike, experiment: utils.MuphysExperiment
) -> None:
    assert experiment.type == utils.ExperimentType.GRAUPEL_ONLY
    inp = common.GraupelInput.load(
        filename=experiment.input_file, allocator=model_backends.get_allocator(backend_like)
    )

    graupel_run_program = run_graupel_only.setup_graupel(
        inp,
        dt=experiment.dt,
        qnc=experiment.qnc,
        backend=backend_like,
        enable_masking=True,  # `False` would require different reference data (or relaxing thresholds)
    )

    # We are passing the same buffers for `Q` as input and output. This is not best GT4Py practice,
    # but save in this case as we are not reading the input with an offset.
    out = common.GraupelOutput.allocate(
        allocator=model_backends.get_allocator(backend_like),
        domain=gtx.domain({dims.CellDim: inp.ncells, dims.KDim: inp.nlev}),
        references={
            "qv": inp.qv,
            "qc": inp.qc,
            "qi": inp.qi,
            "qr": inp.qr,
            "qs": inp.qs,
            "qg": inp.qg,
            "t": inp.t,
        },
    )

    graupel_run_program(
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
