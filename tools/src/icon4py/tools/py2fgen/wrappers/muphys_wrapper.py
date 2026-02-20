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


#   /**
#    * @brief
#    *
#    * @param [in] nvec Number of horizontal points
#    * @param [in] ke Number of grid points in vertical direction
#    * @param [in] ivstart Start index for horizontal direction
#    * @param [in] ivend End index for horizontal direction
#    * @param [in] kstart Start index for vertical direction
#    * @param [in] dt Time step for integration of microphysics (s)
#    * @param [in] dz Layer thickness of full levels (m)
#    * @param [inout] t Temperature in Kelvin
#    * @param [in] rho Density of moist air (kg/m3)
#    * @param [in] p Pressure (Pa)
#    * @param [inout] qv Specific water vapor content (kg/kg)
#    * @param [inout] qc Specific cloud water content (kg/kg)
#    * @param [inout] qi Specific cloud ice content (kg/kg)
#    * @param [inout] qr Specific rain content (kg/kg)
#    * @param [inout] qs Specific snow content  kg/kg)
#    * @param [inout] qg Specific graupel content (kg/kg)
#    * @param [in] qnc Cloud number concentration
#    * @param [out] prr_gsp Precipitation rate of rain, grid-scale (kg/(m2*s))
#    * @param [out] pri_gsp Precipitation rate of ice, grid-scale (kg/(m2*s))
#    * @param [out] prs_gsp Precipitation rate of snow, grid-scale (kg/(m2*s))
#    * @param [out] prg_gsp Precipitation rate of graupel, grid-scale (kg/(m2*s))
#    * @param [out] pre_gsp Energy flux at sfc from precipitation (W/m2)
#    * @param [out] pflx Total precipitation flux
#    *
#    */
#   void run(const mu_int_t nvec, const mu_int_t ke, const mu_int_t ivstart,
#            const mu_int_t ivend, const mu_int_t kstart, const real_t dt,
#            real_t *dz, real_t *t, real_t *rho, real_t *p, real_t *qv,
#            real_t *qc, real_t *qi, real_t *qr, real_t *qs, real_t *qg,
#            const real_t qnc, real_t *prr_gsp, real_t *pri_gsp, real_t *prs_gsp,
#            real_t *prg_gsp, real_t *pflx, real_t *pre_gsp);

graupel_program = None


@icon4py_export.export
def graupel_run(
    # nvec: gtx.int32, #TODO this is probably the number that we have to pass as the horizontal size to our c header
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
    prr_gsp: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    pri_gsp: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    prs_gsp: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    prg_gsp: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    pflx: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    pre_gsp: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
):
    global graupel_program  # noqa: PLW0603
    if graupel_program is None:
        on_gpu = t.array_ns != np
        backend = wrapper_common.select_backend(wrapper_common.BackendIntEnum.DACE, on_gpu=on_gpu)
        with muphys_utils.utils.recursion_limit(10**4):
            graupel_program = model_options.setup_program(
                backend=backend,
                program=graupel.graupel_run,
                constant_args={"dt": dt, "qnc": qnc, "enable_masking": True},
                horizontal_sizes={
                    "horizontal_start": gtx.int32(ivstart),  # TODO double-check these ranges
                    "horizontal_end": gtx.int32(ivend),
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(kstart),  # TODO double-check these ranges
                    "vertical_end": gtx.int32(ke),
                },
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
        pr=prr_gsp,
        ps=prs_gsp,
        pi=pri_gsp,
        pg=prg_gsp,
        pre=pre_gsp,
    )
