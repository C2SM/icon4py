# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers of the tmx integration datatests: state constructors from
the serialized ICON data (exp.exclaim_ape_aesPhys savepoints)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.common import dimension as dims
from icon4py.model.common.metrics import metric_fields


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb


# Serialized timesteps of the exclaim_ape_aesPhys archive (run start
# 2008-09-01T00:00:00Z, dtime = 300 s). The archive also holds the
# 00:00:00 step, but that is the call made during model initialization,
# so the verification tests parametrize over the subsequent steps only.
TMX_DATES: tuple[str, ...] = ("2008-09-01T00:05:00.000", "2008-09-01T00:10:00.000")

# Relative tolerance of all tmx integration datatests, see verify_full_run_fields.
RTOL: float = 3.0e-12


def construct_metric_state(
    *,
    metrics_savepoint: sb.MetricSavepoint,
    init_savepoint: sb.TmxInitSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    allocator: gtx_typing.Allocator | None,
) -> tmx_states.TmxMetricState:
    inv_ddqz_z_full = metrics_savepoint.inv_ddqz_z_full()
    ddqz_z_full = metrics_savepoint.ddqz_z_full()
    if ddqz_z_full is None:  # optionally registered in the savepoint
        ddqz_z_full = gtx.as_field(
            (dims.CellDim, dims.KDim), 1.0 / inv_ddqz_z_full.asnumpy(), allocator=allocator
        )
    z_mc = metrics_savepoint.z_mc()
    z_ifc = metrics_savepoint.z_ifc()
    return tmx_states.TmxMetricState(
        ddqz_z_full=ddqz_z_full,
        inv_ddqz_z_full=inv_ddqz_z_full,
        ddqz_z_half=metrics_savepoint.ddqz_z_half(),
        inv_ddqz_z_half=init_savepoint.inv_ddqz_z_half(),
        inv_ddqz_z_full_e=init_savepoint.inv_ddqz_z_full_e(),
        inv_ddqz_z_half_e=init_savepoint.inv_ddqz_z_half_e(),
        inv_ddqz_z_half_v=init_savepoint.inv_ddqz_z_half_v(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        wgtfac_e=metrics_savepoint.wgtfac_e(),
        wgtfacq_c=metrics_savepoint.wgtfacq_c(),
        wgtfacq1_c=init_savepoint.wgtfacq1_c(),
        wgtfacq_e=metrics_savepoint.wgtfacq_e(),
        wgtfacq1_e=init_savepoint.wgtfacq1_e(),
        geopot_agl_ifc=init_savepoint.geopot_agl_ifc(),
        # as the metrics factory computes it, so the datatest below validates
        # the formula against the serialized 'ghf'
        height_above_ground=gtx.as_field(
            (dims.CellDim, dims.KDim),
            metric_fields.compute_height_above_ground(z_mc=z_mc.ndarray, z_ifc=z_ifc.ndarray),
            allocator=allocator,
        ),
        z_mc=z_mc,
        z_ifc=z_ifc,
        # a grid-geometry field, not part of the common EdgeParams (see the
        # TmxMetricState docstring)
        edge_cell_length=grid_savepoint.edge_cell_length(),
    )


def construct_interpolation_state(
    interpolation_savepoint: sb.InterpolationSavepoint,
) -> tmx_states.TmxInterpolationState:
    return tmx_states.TmxInterpolationState(
        c_lin_e=interpolation_savepoint.c_lin_e(),
        e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
        geofac_div=interpolation_savepoint.geofac_div(),
        # `c_intp` is `p_int_state%cells_aw_verts` in the serialization
        cells_aw_verts=interpolation_savepoint.c_intp(),
        rbf_coeff_v1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_v2=interpolation_savepoint.rbf_vec_coeff_v2(),
        rbf_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
        rbf_coeff_c1=interpolation_savepoint.rbf_vec_coeff_c1(),
        rbf_coeff_c2=interpolation_savepoint.rbf_vec_coeff_c2(),
    )
