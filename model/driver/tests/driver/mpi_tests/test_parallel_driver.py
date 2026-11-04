# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import pathlib
from collections.abc import Mapping

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.common import model_backends, time
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    parallel_helpers,
    test_utils,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    download_ser_data,
    experiment_description,
    process_props,
)


_log = logging.getLogger(__file__)


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "experiment_description, end_of_simulation",
    [
        pytest.param(
            test_defs.Experiments.JW,
            time.NumTimeSteps(1),
            marks=[pytest.mark.level("integration")],
            id="integration-jw",
        ),
        pytest.param(
            test_defs.Experiments.GAUSS3D,
            time.NumTimeSteps(1),
            marks=[pytest.mark.level("integration")],
            id="integration-gauss3d",
        ),
        pytest.param(
            test_defs.Experiments.MCH_CH_R04B09,
            time.NumTimeSteps(1),
            marks=[pytest.mark.level("integration")],
            id="integration-mch-ch-r04b09",
        ),
        pytest.param(
            test_defs.Experiments.EXCLAIM_APE_AES,
            time.NumTimeSteps(1),
            marks=[pytest.mark.level("integration")],
            id="integration-exclaim-ape-aes",
        ),
        pytest.param(
            test_defs.Experiments.JW,
            time.RelativeTime(days=7),
            marks=[pytest.mark.level("validation")],
            id="validation-jw",
        ),
    ],
)
def test_driver_compare_single_multi_rank(  # noqa: PLR0917 [too-many-positional-arguments]
    download_ser_data: None,
    experiment_description: test_defs.ExperimentDescription,
    end_of_simulation: time.EndOfSimulation,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    _run_driver_compare_single_multi_rank(
        experiment_description=experiment_description,
        end_of_simulation=end_of_simulation,
        tmp_path=tmp_path,
        process_props=process_props,
        backend=backend,
    )


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "experiment_description, end_of_simulation",
    [
        pytest.param(
            test_defs.Experiments.EXCLAIM_APE_AES,
            time.NumTimeSteps(1),
            marks=[pytest.mark.level("integration")],
            id="integration-exclaim-ape-aes-tracer-advection",
        ),
        pytest.param(
            test_defs.Experiments.EXCLAIM_APE_AES,
            time.RelativeTime(days=7),
            marks=[pytest.mark.level("validation")],
            id="validation-exclaim-ape-aes-tracer-advection",
        ),
    ],
)
def test_tracer_advection_compare_single_multi_rank(  # noqa: PLR0917 [too-many-positional-arguments]
    download_ser_data: None,
    experiment_description: test_defs.ExperimentDescription,
    end_of_simulation: time.EndOfSimulation,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    """Tracer advection gives bitwise identical results on one and on many ranks.

    The APE experiment advects six tracers with the Miura scheme and the
    positive-definite limiter horizontally and with PPM and the semi-monotonic
    limiter vertically. Microphysics and the physics forcing are switched off
    (ICON's ``iforcing = inone``), so that only the dynamical core, diffusion and
    tracer advection act on the tracers, and diffusion performs the halo exchange
    of w itself instead of leaving it to the physics interface. The water vapour
    of the initial condition is rescaled by a global sum, which is why the global
    reductions have to be order-insensitive for this comparison to be bitwise.
    """
    _run_driver_compare_single_multi_rank(
        experiment_description=experiment_description,
        end_of_simulation=end_of_simulation,
        tmp_path=tmp_path,
        process_props=process_props,
        backend=backend,
        config_overrides={
            "muphys": None,
            "diffusion": {"iforcing": diffusion.ForcingType.NO_FORCING},
        },
    )


def _run_driver_compare_single_multi_rank(
    *,
    experiment_description: test_defs.ExperimentDescription,
    end_of_simulation: time.EndOfSimulation,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    config_overrides: Mapping[str, object] | None = None,
) -> None:
    if experiment_description.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    if model_backends.is_cpu_backend(backend) and test_utils.is_gtfn_backend(backend):
        atol = 1e-12
        rtol = 1e-14
    else:
        atol = 1e-9
        rtol = 0.0
    atol, rtol = test_utils.get_mpi_comparison_tolerance(backend, atol=atol, rtol=rtol)
    if experiment_description is test_defs.Experiments.EXCLAIM_APE_AES:
        # TODO (Yilu): drop this tolerance once #1535 is merged. The multi-rank global sum is
        # order-dependent, so the moisture rescale of the APE initial condition makes qv differ
        # by ~1e-14 relative from step 0, and muphys amplifies that to ~2e-11 in qc, qi and qr.
        # #1535 ports ICON's order-insensitive sum, after which this experiment is bitwise
        # reproducible (verified on #1534). The tolerance still catches a missing halo exchange
        # (1e-5 and more).
        atol, rtol = 1e-16, 1e-9

    _log.info(
        f"running on {process_props.comm} with {process_props.comm_size} ranks and atol = {atol}, rtol = {rtol}"
    )

    allocator = model_backends.get_allocator(backend)

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = driver_config.read_experiment_config_from_yaml(config_file_path / "config.yml")
    if config_overrides is not None:
        config = config.with_overrides(**config_overrides)

    single_rank_process_props = decomp_defs.SingleNodeProcessProperties()
    single_rank_config = config.with_overrides(
        driver={
            "output_path": tmp_path / f"ci_driver_output_serial_rank_{process_props.rank}",
            "end_of_simulation": end_of_simulation,
        }
    )
    single_rank_grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=single_rank_config.vertical_grid,
        allocator=allocator,
        process_props=single_rank_process_props,
    )
    single_rank_ds, _ = driver.run_driver(
        config=single_rank_config,
        grid_manager=single_rank_grid_manager,
        process_props=single_rank_process_props,
        backend=backend,
    )

    multi_rank_config = config.with_overrides(
        driver={
            "output_path": tmp_path / f"ci_driver_output_mpi_rank_{process_props.rank}",
            "end_of_simulation": end_of_simulation,
        }
    )
    multi_rank_grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=multi_rank_config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    multi_rank_ds, multi_rank_driver = driver.run_driver(
        config=multi_rank_config,
        grid_manager=multi_rank_grid_manager,
        process_props=process_props,
        backend=backend,
    )

    prognostics = ["vn", "w", "exner", "theta_v", "rho"]
    single_rank_fields = {
        name: getattr(single_rank_ds.prognostics.current, name) for name in prognostics
    }
    multi_rank_fields = {
        name: getattr(multi_rank_ds.prognostics.current, name) for name in prognostics
    }
    # experiments with tracers compare them as well: stale tracer halos are exactly what
    # the physics driver's end-of-step exchange fixes
    for tracer in single_rank_ds.tracers.current.active_fields():
        single_rank_fields[tracer.name] = tracer.field
    for tracer in multi_rank_ds.tracers.current.active_fields():
        multi_rank_fields[tracer.name] = tracer.field
    single_rank_reference_fields: dict[str, object] = {
        name: field.asnumpy() for name, field in single_rank_fields.items()
    }

    for field_name in single_rank_fields:
        print(f"\nverifying field {field_name}")
        global_reference_field = process_props.comm.bcast(
            single_rank_reference_fields.get(field_name),
            root=0,
        )
        local_field = multi_rank_fields[field_name]
        dim = local_field.domain.dims[0]
        parallel_helpers.check_local_global_field(
            decomposition_info=multi_rank_driver.decomposition_info,
            process_props=process_props,
            dim=dim,
            global_reference_field=global_reference_field,
            local_field=local_field.asnumpy(),
            check_halos=True,
            atol=atol,
            rtol=rtol,
        )
