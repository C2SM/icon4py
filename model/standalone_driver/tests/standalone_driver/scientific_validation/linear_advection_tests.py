# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import pathlib

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.atmosphere.tracer_advection import tracer_advection
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.standalone_driver import config as driver_config, driver_utils, standalone_driver
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    test_utils,
)
from icon4py.model.common.initial_condition.analytical import (
    linear_advection as lin_adv_ic,
)
from icon4py.model.common import initial_condition, prescribed_tendencies
from icon4py.model.common.config import options as common_conf_opt
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.grid.geometry_config import GeometryConfig
from icon4py.model.common.interpolation import interpolation_factory
from icon4py.model.common.metrics import metrics_factory
from icon4py.model.common.states import tracer_states
from icon4py.model.common.topography.analytical import flat_topography as flat_topo
from icon4py.model.standalone_driver import (
    config as driver_config,
)
from ..fixtures import *  # noqa: F403


def _create_vgrid_config_for_horizontal_advection_test(
) -> v_grid.VerticalGridConfig:
    vertical_grid_config = v_grid.VerticalGridConfig(
        num_levels=2,
        model_top_height=1000.0,
        lowest_layer_thickness=0.0,
    )
    return vertical_grid_config


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description, horizontal_advection_type, horizontal_advection_limiter, vertical_advection_type, vertical_advection_limiter",
    [
        (
            test_defs.Experiments.GAUSS3D,
            tracer_advection.HorizontalAdvectionType.LINEAR_2ND_ORDER,
            tracer_advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
            tracer_advection.VerticalAdvectionType.NO_ADVECTION,
            tracer_advection.VerticalAdvectionLimiter.NO_LIMITER,
                        
        ),
    ],
)
def horizontal_advection_test(
    experiment_description,
    horizontal_advection_type,
    horizontal_advection_limiter,
    vertical_advection_type,
    vertical_advection_limiter,
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)

    geometry_config = GeometryConfig()
    metrics_config = metrics_factory.MetricsConfig()
    interpolation_config = interpolation_factory.InterpolationConfig()

    vgrid_config = _create_vgrid_config_for_horizontal_advection_test()
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(
        vertical_config=vgrid_config,
        allocator=allocator,
    )
    topography_config = flat_topo.FlatTopographyConfig()
    initial_condition_config = initial_condition.InitialConditionConfig(
        config=lin_adv_ic.LinearAdvectionConfig(
            tracer_profile=lin_adv_ic.TracerProfile.GAUSS_2D,
            velocity_field=lin_adv_ic.VelocityField.CONSTANT,
        )
    )
    prescribed_tendencies_config = prescribed_tendencies.PrescribedTendenciesConfig(data_path=None)
    start_date = datetime.datetime(1, 1, 1, 0, 0, 0, tzinfo=datetime.UTC)
    driver = driver_config.DriverConfig(
        start_of_simulation=start_date,
        start_of_timestepping=start_date,
        end_of_simulation=start_date + datetime.timedelta(seconds=10),
        dtime=2.0,
        output_path=tmp_path / "ci_driver_output",
    )
    solve_nonhydro_config = None
    diffusion_config = None
    tracer_config = tracer_states.TracerConfig.from_ntracer(1)
    tracer_advection_config = tracer_advection.AdvectionConfig(
        horizontal_advection_type=horizontal_advection_type,
        horizontal_advection_limiter=horizontal_advection_limiter,
        vertical_advection_type=vertical_advection_type,
        vertical_advection_limiter=vertical_advection_limiter,
    )
    graupel_config = None

    experiment_config = driver_config.ExperimentConfig(
        geometry=geometry_config,
        metrics=metrics_config,
        interpolation=interpolation_config,
        vertical_grid=vgrid_config,
        topography=topography_config,
        initial_condition=initial_condition_config,
        prescribed_tendencies=prescribed_tendencies_config,
        driver=driver_config,
        solve_nonhydro=solve_nonhydro_config,
        diffusion=diffusion_config,
        tracer=tracer_config,
        tracer_advection=tracer_advection_config,
        graupel=graupel_config,
    )

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=experiment_config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    ds, _ = standalone_driver.run_driver(
        config=experiment_config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )

    # get reference solution
    tracer_reference = construct_idealized_tracer_reference(
        test_config,
        icon_grid,
        x_center,
        y_center,
        x_range,
        y_range,
        edges_center_x,
        edges_center_y,
        node_x,
        node_y,
        time,
        time_end,
        weights,
        nodes,
        tracer_reference_high,
        cell_center_x_high,
        cell_center_y_high,
    )

    # for name, reference in references.items():
    #     atol, rtol = tolerances[name]
    #     test_utils.assert_dallclose(
    #         computed[name].asnumpy(),
    #         reference.asnumpy(),
    #         atol=atol,
    #         rtol=rtol,
    #         err_msg=name,
    #     )
