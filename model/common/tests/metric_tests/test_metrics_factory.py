# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import (
    metrics_attributes as attrs,
    metrics_factory,
)
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    grid_utils as gridtest_utils,
    helpers as test_helpers,
)


metrics_factories = {}


def metrics_config(experiment: str) -> tuple:
    rayleigh_coeff = 5.0
    lowest_layer_thickness = 50.0
    exner_expol = 0.333
    vwind_offctr = 0.2
    rayleigh_type = 2
    model_top_height = 23500.0
    stretch_factor = 1.0
    damping_height = 45000.0
    match experiment:
        case dt_utils.REGIONAL_EXPERIMENT:
            lowest_layer_thickness = 20.0
            model_top_height = 23000.0
            stretch_factor = 0.65
            damping_height = 12500.0
        case dt_utils.GLOBAL_EXPERIMENT:
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


def get_metrics_factory(
    backend, experiment, grid_file, grid_savepoint, metrics_savepoint
) -> metrics_factory.MetricsFieldsFactory:
    backend_name = test_helpers.extract_backend_name(backend)
    name = experiment.join(backend_name)
    factory = metrics_factories.get(name)

    if not factory:
        geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
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
        interpolation_fact = interpolation_factory.InterpolationFieldsFactory(
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
            interpolation_source=interpolation_fact,
            backend=backend,
            metadata=attrs.attrs,
            interface_model_height=metrics_savepoint.z_ifc(),
            e_refin_ctrl=grid_savepoint.refin_ctrl(dims.EdgeDim),
            c_refin_ctrl=grid_savepoint.refin_ctrl(dims.CellDim),
            damping_height=damping_height,
            rayleigh_type=rayleigh_type,
            rayleigh_coeff=rayleigh_coeff,
            exner_expol=exner_expol,
            vwind_offctr=vwind_offctr,
        )
        metrics_factories[name] = factory
    return factory


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_inv_ddqz_z(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.inv_ddqz_z_full()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.INV_DDQZ_Z_FULL)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy())


@pytest.mark.requires_concat_where
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_ddqz_z_half(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.ddqz_z_half()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.DDQZ_Z_HALF)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy())


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_scalfac_dd3d(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.scalfac_dd3d()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.SCALFAC_DD3D)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy())


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_rayleigh_w(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.rayleigh_w()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.RAYLEIGH_W)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy())


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_coeffs_dwdz(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref_1 = metrics_savepoint.coeff1_dwdz()
    field_ref_2 = metrics_savepoint.coeff2_dwdz()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field_1 = factory.get(attrs.COEFF1_DWDZ)
    field_2 = factory.get(attrs.COEFF2_DWDZ)
    assert test_helpers.dallclose(field_ref_1.asnumpy(), field_1.asnumpy())
    assert test_helpers.dallclose(field_ref_2.asnumpy(), field_2.asnumpy())


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_ref_mc(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref_1 = metrics_savepoint.theta_ref_mc()
    field_ref_2 = metrics_savepoint.exner_ref_mc()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field_1 = factory.get(attrs.THETA_REF_MC)
    field_2 = factory.get(attrs.EXNER_REF_MC)
    assert test_helpers.dallclose(field_ref_1.asnumpy(), field_1.asnumpy())
    assert test_helpers.dallclose(field_ref_2.asnumpy(), field_2.asnumpy())


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_facs_mc(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref_1 = metrics_savepoint.d2dexdz2_fac1_mc()
    field_ref_2 = metrics_savepoint.d2dexdz2_fac2_mc()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field_1 = factory.get(attrs.D2DEXDZ2_FAC1_MC)
    field_2 = factory.get(attrs.D2DEXDZ2_FAC2_MC)
    assert test_helpers.dallclose(field_1.asnumpy(), field_ref_1.asnumpy())
    assert test_helpers.dallclose(field_2.asnumpy(), field_ref_2.asnumpy())


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_ddxn_z_full(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.ddxn_z_full()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.DDXN_Z_FULL)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-8)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (
            dt_utils.REGIONAL_EXPERIMENT,
            dt_utils.REGIONAL_EXPERIMENT,
        ),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.cpu_only
@pytest.mark.datatest
def test_factory_vwind_impl_wgt(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.vwind_impl_wgt()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.VWIND_IMPL_WGT)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-9)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (
            dt_utils.REGIONAL_EXPERIMENT,
            dt_utils.REGIONAL_EXPERIMENT,
        ),  # TODO: check vwind_offctr value for regional
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.cpu_only
@pytest.mark.datatest
def test_factory_vwind_expl_wgt(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.vwind_expl_wgt()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.VWIND_EXPL_WGT)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-8)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_exner_exfac(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.exner_exfac()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.EXNER_EXFAC)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1.0e-5)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_pg_edgeidx_dsl(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.pg_edgeidx_dsl()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.PG_EDGEIDX_DSL)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy())


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_pg_exdist_dsl(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.pg_exdist()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.PG_EDGEDIST_DSL)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), atol=1.0e-5)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_mask_bdy_prog_halo_c(
    grid_savepoint, metrics_savepoint, grid_file, experiment, backend
):
    field_ref_1 = metrics_savepoint.mask_prog_halo_c()
    field_ref_2 = metrics_savepoint.bdy_halo_c()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field_1 = factory.get(attrs.MASK_PROG_HALO_C)
    field_2 = factory.get(attrs.BDY_HALO_C)
    assert test_helpers.dallclose(field_ref_1.asnumpy(), field_1.asnumpy())
    assert test_helpers.dallclose(field_ref_2.asnumpy(), field_2.asnumpy())


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_hmask_dd3d(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.hmask_dd3d()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.HMASK_DD3D)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy())


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.cpu_only
@pytest.mark.datatest
def test_factory_zdiff_gradp(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.zdiff_gradp()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.ZDIFF_GRADP)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), atol=1.0e-5)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_coeff_gradekin(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref = metrics_savepoint.coeff_gradekin()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.COEFF_GRADEKIN)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-8)


@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
    ],
)
@pytest.mark.datatest
def test_factory_wgtfacq_e(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field = factory.get(attrs.WGTFACQ_E)
    field_ref = metrics_savepoint.wgtfacq_e_dsl(field.shape[1])
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy(), rtol=1e-9)


@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
@pytest.mark.cpu_only
@pytest.mark.datatest
def test_factory_diffusion(grid_savepoint, metrics_savepoint, grid_file, experiment, backend):
    field_ref_1 = metrics_savepoint.mask_hdiff()
    field_ref_2 = metrics_savepoint.zd_diffcoef()
    field_ref_3 = metrics_savepoint.zd_intcoef()
    field_ref_4 = metrics_savepoint.zd_vertoffset()
    factory = get_metrics_factory(
        backend=backend,
        experiment=experiment,
        grid_file=grid_file,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
    )
    field_1 = factory.get(attrs.MASK_HDIFF)
    field_2 = factory.get(attrs.ZD_DIFFCOEF_DSL)
    field_3 = factory.get(attrs.ZD_INTCOEF_DSL)
    field_4 = factory.get(attrs.ZD_VERTOFFSET_DSL)
    assert test_helpers.dallclose(field_ref_1.asnumpy(), field_1.asnumpy())
    assert test_helpers.dallclose(field_ref_2.asnumpy(), field_2.asnumpy(), rtol=1.0e-4)
    assert test_helpers.dallclose(field_ref_3.asnumpy(), field_3.asnumpy())
    assert test_helpers.dallclose(field_ref_4.asnumpy(), field_4.asnumpy())
