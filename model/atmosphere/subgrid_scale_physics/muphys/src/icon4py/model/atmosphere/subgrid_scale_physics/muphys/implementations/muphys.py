# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import maximum, minimum, power, sqrt, where
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import (
    _saturation_adjustment
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations.graupel import (
    _graupel_run
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff


@gtx.field_operator
def _muphys_run(
    last_lev: gtx.int32,
    dz: fa.CellKField[ta.wpfloat],
    te: fa.CellKField[ta.wpfloat],  # Temperature
    p: fa.CellKField[ta.wpfloat],  # Pressure
    rho: fa.CellKField[ta.wpfloat],  # Density containing dry air and water constituents
    qve: fa.CellKField[ta.wpfloat],  # Specific humidity
    qce: fa.CellKField[ta.wpfloat],  # Specific cloud water content
    qre: fa.CellKField[ta.wpfloat],  # Specific rain water
    qse: fa.CellKField[ta.wpfloat],  # Specific snow water
    qie: fa.CellKField[ta.wpfloat],  # Specific ice water content
    qge: fa.CellKField[ta.wpfloat],  # Specific graupel water content
    dt: ta.wpfloat,
    qnc: ta.wpfloat,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:

    te, qve, qce = _saturation_adjustment(te, qve, qce, qre, qse, qie, qge, rho)

    te_out, qv_out, qc_out, qr_out, qs_out, qi_out, qg_out, pflx_out, pr_out, ps_out, pi_out, pg_out, pre_out = _graupel_run(dz, te, p, rho, qve, qce, qre, qse, qie, qge, args.dt, args.qnc)

    te_out, qv_out, qc_out = _saturation_adjustment(te_out, qv_out, qc_out, qr_out, qs_out, qi_out, qg_out, rho)

    return te_out, qv_out, qc_out, qr_out, qs_out, qi_out, qg_out, pflx_out, pr_out, ps_out, pi_out, pg_out, pre_out


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def muphys_run(
    last_lev: gtx.int32,
    dz: fa.CellKField[ta.wpfloat],
    te: fa.CellKField[ta.wpfloat],  # Temperature
    p: fa.CellKField[ta.wpfloat],  # Pressure
    rho: fa.CellKField[ta.wpfloat],  # Density containing dry air and water constituents
    qve: fa.CellKField[ta.wpfloat],  # Specific humidityn
    qce: fa.CellKField[ta.wpfloat],  # Specific cloud water content
    qre: fa.CellKField[ta.wpfloat],  # Specific rain water
    qse: fa.CellKField[ta.wpfloat],  # Specific snow water
    qie: fa.CellKField[ta.wpfloat],  # Specific ice water content
    qge: fa.CellKField[ta.wpfloat],  # Specific graupel water content
    dt: ta.wpfloat,  # Time step
    qnc: ta.wpfloat,
    t_out: fa.CellKField[ta.wpfloat],  # Revised temperature
    qv_out: fa.CellKField[ta.wpfloat],  # Revised humidity
    qc_out: fa.CellKField[ta.wpfloat],  # Revised cloud water
    qr_out: fa.CellKField[ta.wpfloat],  # Revised rain water
    qs_out: fa.CellKField[ta.wpfloat],  # Revised snow water
    qi_out: fa.CellKField[ta.wpfloat],  # Revised ice water
    qg_out: fa.CellKField[ta.wpfloat],  # Revised graupel water
    pflx: fa.CellKField[ta.wpfloat],  # Total precipitation flux
    pr: fa.CellKField[ta.wpfloat],  # Precipitation of rain
    ps: fa.CellKField[ta.wpfloat],  # Precipitation of snow
    pi: fa.CellKField[ta.wpfloat],  # Precipitation of ice
    pg: fa.CellKField[ta.wpfloat],  # Precipitation of graupel
    pre: fa.CellKField[ta.wpfloat],  # Precipitation of graupel
):
    _muphys_run(
        last_lev,
        dz,
        te,
        p,
        rho,
        qve,
        qce,
        qre,
        qse,
        qie,
        qge,
        dt,
        qnc,
        out=(t_out, qv_out, qc_out, qr_out, qs_out, qi_out, qg_out, pflx, pr, ps, pi, pg, pre),
    )
