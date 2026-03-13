# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime
import pathlib

from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.model.common import type_alias as ta


@dataclasses.dataclass
class ProfilingStats:
    gt4py_metrics_level: int = gtx_metrics.ALL
    gt4py_metrics_output_file: str = "gt4py_metrics.json"
    skip_first_timestep: bool = True


@dataclasses.dataclass(frozen=True)
class DriverConfig:
    """
    Standalone driver configuration.

    Default values should correspond to default values in ICON.
    """

    experiment_name: str
    output_path: pathlib.Path
    profiling_stats: ProfilingStats | None
    dtime: datetime.timedelta = datetime.timedelta(seconds=600.0)
    start_date: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0, 0)
    end_date: datetime.datetime = datetime.datetime(1, 1, 1, 1, 0, 0)
    apply_extra_second_order_divdamp: bool = False
    vertical_cfl_threshold: ta.wpfloat = 0.85
    ndyn_substeps: int = 5
    enable_statistics_output: bool = False
    ntracer: int = 0
