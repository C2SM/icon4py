# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.metrics import metrics_attributes as attrs, metrics_factory
from icon4py.model.testing import definitions as test_defs, parallel_helpers, test_utils

from ...fixtures import (
    backend,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    geometry_from_savepoint,
    grid_savepoint,
    icon_grid,
    interpolation_factory_from_savepoint,
    metrics_factory_from_savepoint,
    metrics_savepoint,
    processor_props,
    ranked_data_path,
    topography_savepoint,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, metrics_name",
    [
        (attrs.CELL_HEIGHT_ON_HALF_LEVEL, "z_ifc"),
        (attrs.DDQZ_Z_FULL_E, "ddqz_z_full_e"),
        (attrs.ZDIFF_GRADP, "zdiff_gradp"),
        (attrs.VERTOFFSET_GRADP, "vertoffset_gradp"),
        (attrs.Z_MC, "z_mc"),
        (attrs.DDQZ_Z_HALF, "ddqz_z_half"),
        (attrs.SCALING_FACTOR_FOR_3D_DIVDAMP, "scalfac_dd3d"),
        (attrs.RAYLEIGH_W, "rayleigh_w"),
        (attrs.COEFF_GRADEKIN, "coeff_gradekin"),
    ],
)
def test_distributed_metrics_attrs(
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    metrics_factory_from_savepoint: metrics_factory.MetricsFieldsFactory,
    attrs_name: str,
    metrics_name: str,
    experiment: test_defs.Experiment,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = metrics_factory_from_savepoint
    print(f"computed flatlev {factory.vertical_grid.nflatlev}, expected{grid_savepoint.nflatlev()}")

    field = factory.get(attrs_name).asnumpy()
    field_ref = metrics_savepoint.__getattribute__(metrics_name)().asnumpy()
    assert test_utils.dallclose(field, field_ref, rtol=1e-8, atol=1.0e-8)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, metrics_name",
    [
        (attrs.DDQZ_Z_FULL, "ddqz_z_full"),
        (attrs.INV_DDQZ_Z_FULL, "inv_ddqz_z_full"),
        (attrs.COEFF1_DWDZ, "coeff1_dwdz"),
        (attrs.COEFF2_DWDZ, "coeff2_dwdz"),
        (attrs.THETA_REF_MC, "theta_ref_mc"),
        (attrs.EXNER_REF_MC, "exner_ref_mc"),
        (attrs.RHO_REF_ME, "rho_ref_me"),
        (attrs.THETA_REF_ME, "theta_ref_me"),
        (attrs.D2DEXDZ2_FAC1_MC, "d2dexdz2_fac1_mc"),
        (attrs.D2DEXDZ2_FAC2_MC, "d2dexdz2_fac2_mc"),
        (attrs.DDXN_Z_FULL, "ddxn_z_full"),
        (attrs.DDXT_Z_FULL, "ddxt_z_full"),
        (attrs.EXNER_W_IMPLICIT_WEIGHT_PARAMETER, "vwind_impl_wgt"),
        (attrs.EXNER_W_EXPLICIT_WEIGHT_PARAMETER, "vwind_expl_wgt"),
        (attrs.PG_EDGEDIST_DSL, "pg_exdist"),
        (attrs.PG_EDGEIDX_DSL, "pg_edgeidx_dsl"),
        (attrs.MASK_PROG_HALO_C, "mask_prog_halo_c"),
        (attrs.BDY_HALO_C, "bdy_halo_c"),
        (attrs.HORIZONTAL_MASK_FOR_3D_DIVDAMP, "hmask_dd3d"),
        (attrs.WGTFAC_C, "wgtfac_c"),
        (attrs.EXNER_EXFAC, "exner_exfac"),
    ],
)
def test_distributed_metrics_attrs_no_halo(
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    metrics_factory_from_savepoint: metrics_factory.MetricsFieldsFactory,
    attrs_name: str,
    metrics_name: str,
    experiment: test_defs.Experiment,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = metrics_factory_from_savepoint

    field = factory.get(attrs_name).asnumpy()
    field_ref = metrics_savepoint.__getattribute__(metrics_name)().asnumpy()
    assert test_utils.dallclose(field, field_ref, rtol=1e-7, atol=1.0e-8)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, metrics_name",
    [
        (attrs.MASK_HDIFF, "mask_hdiff"),
        (attrs.ZD_DIFFCOEF_DSL, "zd_diffcoef"),
        (attrs.ZD_INTCOEF_DSL, "zd_intcoef"),
        (attrs.ZD_VERTOFFSET_DSL, "zd_vertoffset"),
    ],
)
def test_distributed_metrics_attrs_no_halo_regional(
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    metrics_factory_from_savepoint: metrics_factory.MetricsFieldsFactory,
    attrs_name: str,
    metrics_name: str,
    experiment: test_defs.Experiment,
) -> None:
    if experiment == test_defs.Experiments.EXCLAIM_APE:
        pytest.skip(f"Fields not computed for {experiment}")
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = metrics_factory_from_savepoint

    field = factory.get(attrs_name).asnumpy()
    field_ref = metrics_savepoint.__getattribute__(metrics_name)().asnumpy()
    assert test_utils.dallclose(field, field_ref, atol=1e-8)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_distributed_metrics_wgtfacq_e(
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    metrics_factory_from_savepoint: metrics_factory.MetricsFieldsFactory,
    experiment: test_defs.Experiment,
) -> None:
    parallel_helpers.check_comm_size(processor_props)
    parallel_helpers.log_process_properties(processor_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = metrics_factory_from_savepoint

    field = factory.get(attrs.WGTFACQ_E).asnumpy()
    field_ref = metrics_savepoint.wgtfacq_e_dsl(field.shape[1]).asnumpy()
    assert test_utils.dallclose(field, field_ref)
