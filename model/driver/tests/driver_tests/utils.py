# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime, timedelta

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.driver import icon4py_configuration as driver_config


# TODO (Chia Rui): The diffusion and solve_nonhydro configs are duplication of the same functions in
#                  diffusion_test/utils.py and dycore_tests/utils.py. Remove the duplication when
#                  architecture for configuration is set up.


def exclaim_ape_diffusion_config(ndyn_substeps):
    """Create DiffusionConfig matching EXCLAIM_APE_R04B02.

    Set values to the ones used in the  EXCLAIM_APE_R04B02 experiment where they differ
    from the default.
    """
    return diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        zdiffu_t=False,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        smagorinski_scaling_factor=0.025,
        hdiff_temp=True,
        n_substeps=ndyn_substeps,
    )


def r04b09_diffusion_config(
    ndyn_substeps,  # imported `ndyn_substeps` fixture
) -> diffusion.DiffusionConfig:
    """
    Create DiffusionConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=True,
        thslp_zdiffu=0.02,
        thhgtd_zdiffu=125.0,
        velocity_boundary_diffusion_denom=150.0,
        max_nudging_coeff=0.075,
        n_substeps=ndyn_substeps,
        shear_type=diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
    )


def construct_diffusion_config(name: str, ndyn_substeps: int = 5):
    if name.lower() in "mch_ch_r04b09_dsl":
        return r04b09_diffusion_config(ndyn_substeps)
    elif name.lower() in "exclaim_ape_r02b04":
        return exclaim_ape_diffusion_config(ndyn_substeps)


def mch_ch_r04b09_dsl_nonhydrostatic_config(ndyn_substeps):
    """Create configuration matching the mch_chR04b09_dsl experiment."""
    config = solve_nh.NonHydrostaticConfig(
        ndyn_substeps_var=ndyn_substeps,
        divdamp_order=solve_nh.DivergenceDampingOrder.COMBINED,
        iau_wgt_dyn=1.0,
        divdamp_fac=0.004,
        max_nudging_coeff=0.075,
    )
    return config


def exclaim_ape_nonhydrostatic_config(ndyn_substeps):
    """Create configuration for EXCLAIM APE experiment."""
    return solve_nh.NonHydrostaticConfig(
        rayleigh_coeff=0.1,
        divdamp_order=solve_nh.DivergenceDampingOrder.COMBINED,
        ndyn_substeps_var=ndyn_substeps,
    )


def construct_nonhydrostatic_config(name: str, ndyn_substeps: int = 5):
    if name.lower() in "mch_ch_r04b09_dsl":
        return mch_ch_r04b09_dsl_nonhydrostatic_config(ndyn_substeps)
    elif name.lower() in "exclaim_ape_r02b04":
        return exclaim_ape_nonhydrostatic_config(ndyn_substeps)


def mch_ch_r04b09_dsl_icon4pyrun_config(
    date_init: str,
    date_exit: str,
    diffusion_linit_init: bool,
    ndyn_substeps: int,
) -> driver_config.Icon4pyRunConfig:
    """
    Create Icon4pyRunConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return driver_config.Icon4pyRunConfig(
        dtime=timedelta(seconds=10.0),
        start_date=datetime.fromisoformat(date_init),
        end_date=datetime.fromisoformat(date_exit),
        n_substeps=ndyn_substeps,
        apply_initial_stabilization=True,
        restart_mode=not diffusion_linit_init,
    )


def exclaim_ape_icon4pyrun_config(
    date_init: str,
    date_exit: str,
    diffusion_linit_init: bool,
    ndyn_substeps: int,
) -> driver_config.Icon4pyRunConfig:
    """
    Create Icon4pyRunConfig matching exclaim_ape_R02B04.

    Set values to the ones used in the exclaim_ape_R02B04 experiment where they differ
    from the default.
    """
    return driver_config.Icon4pyRunConfig(
        dtime=timedelta(seconds=2.0),
        start_date=datetime.fromisoformat(date_init),
        end_date=datetime.fromisoformat(date_exit),
        n_substeps=ndyn_substeps,
        apply_initial_stabilization=False,
        restart_mode=not diffusion_linit_init,
    )


def construct_icon4pyrun_config(
    name: str,
    date_init: str,
    date_exit: str,
    diffusion_linit_init: bool,
    ndyn_substeps: int = 5,
):
    if name.lower() in "mch_ch_r04b09_dsl":
        return mch_ch_r04b09_dsl_icon4pyrun_config(
            date_init, date_exit, diffusion_linit_init, ndyn_substeps
        )
    elif name.lower() in "exclaim_ape_r02b04":
        return exclaim_ape_icon4pyrun_config(
            date_init, date_exit, diffusion_linit_init, ndyn_substeps
        )
