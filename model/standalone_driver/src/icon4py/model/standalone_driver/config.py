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
from typing import Any

from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.model.common import type_alias as ta
from icon4py.model.common.utils.fortran_config import list_to_value


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
    profiling_stats: ProfilingStats | None
    output_path: pathlib.Path = dataclasses.field(default_factory=lambda: pathlib.Path("./output"))
    dtime: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime
    apply_extra_second_order_divdamp: bool = False
    vertical_cfl_threshold: ta.wpfloat = dataclasses.field(default_factory=lambda: ta.wpfloat(0.85))
    ndyn_substeps: int = 5
    enable_statistics_output: bool = False
    ntracer: int = 0

    @classmethod
    def from_fortran_dict(
        cls, atmo_dict: dict[str, Any], master_dict: dict[str, Any], **overrides: Any
    ) -> "DriverConfig":
        nonhydrostatic_nml = atmo_dict["nonhydrostatic_nml"]
        run_nml = atmo_dict["run_nml"]
        master_time_control_nml = master_dict["master_time_control_nml"]
        dtime = run_nml["dtime"]
        start_date_str = master_time_control_nml["experimentstartdate"]
        end_date_str = master_time_control_nml["experimentstopdate"]
        return cls(
            dtime=datetime.timedelta(seconds=dtime),
            start_date=datetime.datetime.fromisoformat(start_date_str.replace("Z", "+00:00")),
            end_date=datetime.datetime.fromisoformat(end_date_str.replace("Z", "+00:00")),
            apply_extra_second_order_divdamp=nonhydrostatic_nml["lextra_diffu"],
            vertical_cfl_threshold=ta.wpfloat(str(nonhydrostatic_nml["vcfl_threshold"])),
            ndyn_substeps=nonhydrostatic_nml["ndyn_substeps"],
            ntracer=list_to_value(run_nml["ntracer"]),
            **overrides,
        )
