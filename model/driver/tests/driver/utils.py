# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime, timedelta

import gt4py.next.typing as gtx_typing

from icon4py.model.driver import icon4py_configuration as driver_config
from icon4py.model.testing import definitions


def mch_ch_r04b09_dsl_icon4pyrun_config(
    date_init: str,
    date_exit: str,
    diffusion_linit_init: bool,
    backend: gtx_typing.Backend,
    ndyn_substeps: int,
) -> driver_config.Icon4pyRunConfig:
    """
    Create Icon4pyRunConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default. Backend is not used because granules are set independently in test_timeloop.py.
    """
    return driver_config.Icon4pyRunConfig(
        dtime=timedelta(seconds=10.0),
        start_date=datetime.fromisoformat(date_init),
        end_date=datetime.fromisoformat(date_exit),
        n_substeps=ndyn_substeps,
        apply_initial_stabilization=True,
        restart_mode=not diffusion_linit_init,
        backend=backend,
    )


def exclaim_ape_icon4pyrun_config(
    date_init: str,
    date_exit: str,
    diffusion_linit_init: bool,
    backend: gtx_typing.Backend,
    ndyn_substeps: int,
) -> driver_config.Icon4pyRunConfig:
    """
    Create Icon4pyRunConfig matching exclaim_ape_R02B04.

    Set values to the ones used in the exclaim_ape_R02B04 experiment where they differ
    from the default. Backend is not used because granules are set independently in test_timeloop.py
    """
    return driver_config.Icon4pyRunConfig(
        dtime=timedelta(seconds=2.0),
        start_date=datetime.fromisoformat(date_init),
        end_date=datetime.fromisoformat(date_exit),
        n_substeps=ndyn_substeps,
        apply_initial_stabilization=False,
        restart_mode=not diffusion_linit_init,
        backend=backend,
    )


def construct_icon4pyrun_config(
    experiment: definitions.Experiment,
    date_init: str,
    date_exit: str,
    diffusion_linit_init: bool,
    backend: gtx_typing.Backend,
    ndyn_substeps: int = 5,
):
    if experiment == definitions.Experiments.MCH_CH_R04B09:
        return mch_ch_r04b09_dsl_icon4pyrun_config(
            date_init, date_exit, diffusion_linit_init, backend, ndyn_substeps
        )
    elif experiment == definitions.Experiments.EXCLAIM_APE:
        return exclaim_ape_icon4pyrun_config(
            date_init, date_exit, diffusion_linit_init, backend, ndyn_substeps
        )
