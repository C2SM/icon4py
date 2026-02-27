# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from gt4py import next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import utils as muphys_utils
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations import graupel
from icon4py.model.common import dimension as dims, model_options
from icon4py.tools.py2fgen.wrappers import common as wrapper_common, icon4py_export


graupel_program = None


@icon4py_export.export
def graupel_run(
    ke: gtx.int32,
    ivstart: gtx.int32,
    ivend: gtx.int32,
    kstart: gtx.int32,
    dt: gtx.float64,
    dz: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    t: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    rho: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    p: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    qv: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    qc: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    qi: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    qr: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    qs: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    qg: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    qnc: gtx.float64,
    prr_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    pri_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    prs_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    prg_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    pflx: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    pre_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
):
    global graupel_program  # noqa: PLW0603 [global-statement]
    if graupel_program is None:
        on_gpu = t.array_ns != np
        backend = wrapper_common.select_backend(wrapper_common.BackendIntEnum.DACE, on_gpu=on_gpu)
        with muphys_utils.recursion_limit(10**4):
            graupel_program = model_options.setup_program(
                backend=backend,
                program=graupel.graupel_run,
                constant_args={"dt": dt, "qnc": qnc, "enable_masking": True},
                horizontal_sizes={
                    "horizontal_start": gtx.int32(ivstart),
                    "horizontal_end": gtx.int32(ivend),
                },  # TODO(edopao): double-check these ranges
                vertical_sizes={
                    "vertical_start": gtx.int32(kstart),
                    "vertical_end": gtx.int32(ke),
                },  # TODO(edopao): double-check these ranges
                offset_provider={"Koff": dims.KDim},
            )
            gtx.wait_for_compilation()

    q = graupel.Q(qv, qc, qi, qr, qs, qg)

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
