# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
import logging
import math
import os
from cmath import sqrt as c_sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from icon4py.model.common.config import Device
from icon4py.model.common.decomposition.definitions import (
    get_processor_properties,
    get_runtype,
)
from icon4py.model.common.settings import device, xp
from icon4py.model.driver.dycore_driver import initialize
from icon4py.model.driver.initialization_utils import (
    WAVETYPE,
    DivWave,
    ExperimentType,
    SerializationType,
    configure_logging,
    determine_u_v_w_in_div_converge_experiment,
    read_geometry_fields,
    read_static_fields,
)
from icon4py.model.driver import divergence_utils as div_utils


log = logging.getLogger(__name__)



def test_jabw():
    resolutions = (
        "4",
        "5",
        '6',
    )
    grid_level = (4, 5, 6)
    dtime_seconds = (600.0, 300.0, 150.0)
    
    base_input_path = "/scratch/mch/cong/data/nh35_tri_jws_sb_exclaim_r2b"
    run_path = "./"
    experiment_type = ExperimentType.JABW
    serialization_type = SerializationType.SB
    
    disable_logging = False
    mpi = False
    profile = False
    grid_root = 2
    enable_output = True
    enable_debug_message = False

    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + "/ser_data/"
        parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
        configure_logging(run_path, experiment_type, parallel_props, disable_logging)

        (
            timeloop,
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            diagnostic_metric_state,
            diagnostic_state,
            prognostic_state_list,
            output_state,
            prep_adv,
            inital_divdamp_fac_o2,
        ) = initialize(
            Path(input_path),
            parallel_props,
            serialization_type,
            experiment_type,
            grid_root,
            grid_level[i],
            enable_output,
            enable_debug_message,
            div_wave=None,
            dtime_seconds=dtime_seconds[i],
        )

        timeloop.time_integration(
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            diagnostic_metric_state,
            diagnostic_state,
            prognostic_state_list,
            prep_adv,
            inital_divdamp_fac_o2,
            do_prep_adv=False,
            output_state=output_state,
            profile=profile,
        )

        os.system(f"mkdir -p data_r2b{res}")
        os.system(f"mv data_output* dummy* nohup* data_r2b{res}/")


if __name__ == "__main__":
    test_jabw()
