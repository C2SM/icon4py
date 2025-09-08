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
from gt4py.next import backend as gtx_backend

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import (
    metrics_attributes as attrs,
    metrics_factory,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    definitions,
    grid_utils as gridtest_utils,
    serialbox,
    test_utils as test_helpers,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    grid_savepoint,
    icon_grid,
    metrics_savepoint,
    processor_props,
    ranked_data_path,
    topography_savepoint,
)

if TYPE_CHECKING:
    from icon4py.model.common.grid import base as base_grid


# TODO(havogt): use everywhere
@pytest.fixture(params=[definitions.Experiments.MCH_CH_R04B09, definitions.Experiments.EXCLAIM_APE])
def experiment(request: pytest.FixtureRequest) -> definitions.Experiment:
    return request.param


metrics_factories: dict[str, metrics_factory.MetricsFieldsFactory] = {}


def metrics_config(experiment: definitions.Experiment) -> tuple:
    rayleigh_coeff = 5.0
    lowest_layer_thickness = 50.0
    exner_expol = 0.333
    vwind_offctr = 0.2
    rayleigh_type = 2
    model_top_height = 23500.0
    stretch_factor = 1.0
    damping_height = 45000.0
    match experiment:
        case definitions.Experiments.MCH_CH_R04B09:
            lowest_layer_thickness = 20.0
            model_top_height = 23000.0
            stretch_factor = 0.65
            damping_height = 12500.0
        case definitions.Experiments.EXCLAIM_APE:
            model_top_height = 75000.0
            stretch_factor = 0.9
            damping_height = 50000.0
            rayleigh_coeff = 0.1
            exner_expol = 0.3333333333333
            vwind_offctr = 0.15

    return (
        lowest_layer_thickness,
        model_top_height,
        stretch_factor,
        damping_height,
        rayleigh_coeff,
        exner_expol,
        vwind_offctr,
        rayleigh_type,
    )


def _get_metrics_factory(
    backend: gtx_backend.Backend | None,
    experiment: definitions.Experiment,
    grid_savepoint: serialbox.IconGridSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
) -> metrics_factory.MetricsFieldsFactory:
    registry_name = "_".join((experiment.name, data_alloc.backend_name(backend)))
    factory = metrics_factories.get(registry_name)

    topography = topography_savepoint.topo_c()

    if not factory:
        geometry = gridtest_utils.get_grid_geometry(backend, experiment)
        (
            lowest_layer_thickness,
            model_top_height,
            stretch_factor,
            damping_height,
            rayleigh_coeff,
            exner_expol,
            vwind_offctr,
            rayleigh_type,
        ) = metrics_config(experiment)

        vertical_config = v_grid.VerticalGridConfig(
            geometry.grid.num_levels,
            lowest_layer_thickness=lowest_layer_thickness,
            model_top_height=model_top_height,
            stretch_factor=stretch_factor,
            rayleigh_damping_height=damping_height,
        )
        vertical_grid = v_grid.VerticalGrid(
            vertical_config, grid_savepoint.vct_a(), grid_savepoint.vct_b()
        )
        interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
            grid=geometry.grid,
            decomposition_info=geometry._decomposition_info,
            geometry_source=geometry,
            backend=backend,
            metadata=interpolation_attributes.attrs,
        )
        factory = metrics_factory.MetricsFieldsFactory(
            grid=geometry.grid,
            vertical_grid=vertical_grid,
            decomposition_info=geometry._decomposition_info,
            geometry_source=geometry,
            topography=topography,
            interpolation_source=interpolation_field_source,
            backend=backend,
            metadata=attrs.attrs,
            e_refin_ctrl=grid_savepoint.refin_ctrl(dims.EdgeDim),
            c_refin_ctrl=grid_savepoint.refin_ctrl(dims.CellDim),
            rayleigh_type=rayleigh_type,
            rayleigh_coeff=rayleigh_coeff,
            exner_expol=exner_expol,
            vwind_offctr=vwind_offctr,
        )
        metrics_factories[registry_name] = factory
    return factory


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_z_mc(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.z_mc()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.Z_MC)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-10)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_ddqz_z_and_inverse(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    inverse_field_ref = metrics_savepoint.inv_ddqz_z_full()
    field_ref = metrics_savepoint.ddqz_z_full()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    inverse_field = factory.get(attrs.INV_DDQZ_Z_FULL)
    field = factory.get(attrs.DDQZ_Z_FULL)
    assert test_helpers.dallclose(inverse_field_ref.asnumpy(), inverse_field.asnumpy(), atol=1e-10)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-7)


@pytest.mark.datatest
def test_factory_ddqz_full_e(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.ddqz_z_full_e().asnumpy()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.DDQZ_Z_FULL_E)
    assert test_helpers.dallclose(field_ref, field.asnumpy(), rtol=1e-8)


@pytest.mark.level("integration")
@pytest.mark.datatest
@pytest.mark.uses_concat_where
def test_factory_ddqz_z_half(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.ddqz_z_half()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.DDQZ_Z_HALF)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-9)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_scaling_factor_for_3d_divdamp(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.scalfac_dd3d()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.SCALING_FACTOR_FOR_3D_DIVDAMP)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy())


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_rayleigh_w(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.rayleigh_w()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.RAYLEIGH_W)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy())


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_coeffs_dwdz(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref_1 = metrics_savepoint.coeff1_dwdz()
    field_ref_2 = metrics_savepoint.coeff2_dwdz()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field_1 = factory.get(attrs.COEFF1_DWDZ)
    field_2 = factory.get(attrs.COEFF2_DWDZ)
    assert test_helpers.dallclose(field_ref_1.asnumpy(), field_1.asnumpy(), atol=1e-11)
    assert test_helpers.dallclose(field_ref_2.asnumpy(), field_2.asnumpy(), atol=1e-11)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_ref_mc(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref_1 = metrics_savepoint.theta_ref_mc()
    field_ref_2 = metrics_savepoint.exner_ref_mc()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field_1 = factory.get(attrs.THETA_REF_MC)
    field_2 = factory.get(attrs.EXNER_REF_MC)
    assert test_helpers.dallclose(field_ref_1.asnumpy(), field_1.asnumpy(), atol=1e-9)
    assert test_helpers.dallclose(field_ref_2.asnumpy(), field_2.asnumpy(), atol=1e-10)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_d2dexdz2_facs_mc(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref_1 = metrics_savepoint.d2dexdz2_fac1_mc()
    field_ref_2 = metrics_savepoint.d2dexdz2_fac2_mc()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field_1 = factory.get(attrs.D2DEXDZ2_FAC1_MC)
    field_2 = factory.get(attrs.D2DEXDZ2_FAC2_MC)
    assert test_helpers.dallclose(field_1.asnumpy(), field_ref_1.asnumpy(), atol=1e-12)
    assert test_helpers.dallclose(field_2.asnumpy(), field_ref_2.asnumpy(), atol=1e-12)


@pytest.mark.datatest
def test_factory_ddxn_z_full(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.ddxn_z_full()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.DDXN_Z_FULL)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), atol=1e-8)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_ddxt_z_full(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.ddxt_z_full().asnumpy()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.DDXT_Z_FULL)
    # TODO(halungge): these are the np.allclose default values: single precision
    assert test_helpers.dallclose(field.asnumpy(), field_ref, rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_exner_w_implicit_weight_parameter(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.vwind_impl_wgt()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.EXNER_W_IMPLICIT_WEIGHT_PARAMETER)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-9)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_exner_w_explicit_weight_parameter(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.vwind_expl_wgt()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.EXNER_W_EXPLICIT_WEIGHT_PARAMETER)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-8)


@pytest.mark.level("integration")
@pytest.mark.uses_concat_where
@pytest.mark.datatest
def test_factory_exner_exfac(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.exner_exfac()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.EXNER_EXFAC)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), atol=1e-8)


@pytest.mark.level("integration")
@pytest.mark.embedded_remap_error
@pytest.mark.datatest
def test_factory_pressure_gradient_fields(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_1_ref = metrics_savepoint.pg_exdist()
    field_2_ref = metrics_savepoint.pg_edgeidx_dsl()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field_1 = factory.get(attrs.PG_EDGEDIST_DSL)
    assert test_helpers.dallclose(field_1_ref.asnumpy(), field_1.asnumpy(), atol=1.0e-5)
    field_2 = factory.get(attrs.PG_EDGEIDX_DSL)
    assert test_helpers.dallclose(field_2_ref.asnumpy(), field_2.asnumpy())


@pytest.mark.datatest
def test_factory_mask_bdy_prog_halo_c(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref_1 = metrics_savepoint.mask_prog_halo_c()
    field_ref_2 = metrics_savepoint.bdy_halo_c()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field_1 = factory.get(attrs.MASK_PROG_HALO_C)
    field_2 = factory.get(attrs.BDY_HALO_C)
    assert test_helpers.dallclose(field_ref_1.asnumpy(), field_1.asnumpy())
    assert test_helpers.dallclose(field_ref_2.asnumpy(), field_2.asnumpy())


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_horizontal_mask_for_3d_divdamp(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.hmask_dd3d()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.HORIZONTAL_MASK_FOR_3D_DIVDAMP)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy())


@pytest.mark.level("integration")
@pytest.mark.embedded_remap_error
@pytest.mark.cpu_only  # TODO(halungge): slow on GPU due to vwind_impl_wgt computation)
@pytest.mark.datatest
def test_factory_zdiff_gradp(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.zdiff_gradp()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.ZDIFF_GRADP)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), atol=1.0e-5)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_coeff_gradekin(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref = metrics_savepoint.coeff_gradekin()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.COEFF_GRADEKIN)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-8)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_factory_wgtfacq_e(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.WGTFACQ_E)
    field_ref = metrics_savepoint.wgtfacq_e_dsl(field.shape[1])
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-9)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_vertical_coordinates_on_half_levels(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field = factory.get(attrs.CELL_HEIGHT_ON_HALF_LEVEL)
    field_ref = metrics_savepoint.z_ifc()
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-9)


@pytest.mark.level("integration")
@pytest.mark.embedded_remap_error
@pytest.mark.datatest
def test_factory_compute_diffusion_metrics(
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    experiment: definitions.Experiment,
    backend: gtx_backend.Backend | None,
):
    field_ref_1 = metrics_savepoint.mask_hdiff()
    field_ref_2 = metrics_savepoint.zd_diffcoef()
    field_ref_3 = metrics_savepoint.zd_intcoef()
    field_ref_4 = metrics_savepoint.zd_vertoffset()
    factory = _get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_savepoint=grid_savepoint,
        topography_savepoint=topography_savepoint,
    )
    field_1 = factory.get(attrs.MASK_HDIFF)
    field_2 = factory.get(attrs.ZD_DIFFCOEF_DSL)
    field_3 = factory.get(attrs.ZD_INTCOEF_DSL)
    field_4 = factory.get(attrs.ZD_VERTOFFSET_DSL)
    assert test_helpers.dallclose(field_ref_1.asnumpy(), field_1.asnumpy())
    assert test_helpers.dallclose(field_ref_2.asnumpy(), field_2.asnumpy(), atol=1.0e-10)
    assert test_helpers.dallclose(field_ref_3.asnumpy(), field_3.asnumpy(), atol=1.0e-8)
    assert test_helpers.dallclose(field_ref_4.asnumpy(), field_4.asnumpy())
