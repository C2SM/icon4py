# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from gt4py import next as gtx
from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import utils as muphys_utils
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations import graupel
from icon4py.model.common import dimension as dims, model_backends, model_options, type_alias as ta
from icon4py.tools.py2fgen.wrappers import icon4py_export


graupel_program = None


@icon4py_export.export
def graupel_run(
    ke: gtx.int32,
    ivstart: gtx.int32,
    ivend: gtx.int32,
    kstart: gtx.int32,
    dt: ta.wpfloat,
    dz: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    t: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    rho: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    p: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qv: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qc: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qi: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qr: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qs: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qg: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qnc: ta.wpfloat,
    prr_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    pri_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    prs_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    prg_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    pflx: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    pre_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    wait_for_completion: bool,
):
    global graupel_program  # noqa: PLW0603 [global-statement]
    if graupel_program is None:
        on_gpu = t.array_ns != np
        with muphys_utils.recursion_limit(10**4):
            graupel_program = model_options.setup_program(
                backend={
                    "backend_factory": model_backends.make_custom_dace_backend,
                    "device": model_backends.GPU if on_gpu else model_backends.CPU,
                    "async_sdfg_call": not wait_for_completion,
                },
                program=graupel.graupel_run,
                constant_args={
                    "dt": ta.wpfloat(dt),
                    "qnc": ta.wpfloat(qnc),
                    "enable_masking": True,
                },
                horizontal_sizes={
                    "horizontal_start": gtx.int32(ivstart),
                    "horizontal_end": gtx.int32(ivend),
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(kstart),
                    "vertical_end": gtx.int32(ke),
                },
                offset_provider={"Koff": dims.KDim},
            )
            gtx.wait_for_compilation()

    q = graupel.Q(qv, qc, qr, qs, qi, qg)

    # The precipitation fields (pr, ps, pi, pg, pre) are defined as 1D-fields with
    # horizontal domain, in the driver, because they represent precipitation at the
    # surface level. The graupel program requires a 2D-domain, with a single vertical
    # level corresponding to the surface level. The vertcal shift below is to set
    # the field origin at the surface level, as required by the program.
    graupel_program(
        dz=dz,
        te=t,
        p=p,
        rho=rho,
        q_in=q,
        t_out=t,
        q_out=q,
        pflx=pflx,
        pr=prr_gsp(dims.KDim - (ke - 1)),
        ps=prs_gsp(dims.KDim - (ke - 1)),
        pi=pri_gsp(dims.KDim - (ke - 1)),
        pg=prg_gsp(dims.KDim - (ke - 1)),
        pre=pre_gsp(dims.KDim - (ke - 1)),
    )


@icon4py_export.export
def graupel_finalize():
    # The atexit function is not called when embedding cpython into another application
    # with cffi, so we call it explicitly here to dump the metrics.
    gtx_metrics._dump_metrics_at_exit()
