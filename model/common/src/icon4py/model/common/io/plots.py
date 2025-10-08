# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import pickle


# Prevent matplotlib logging spam
logging.getLogger("matplotlib").setLevel(logging.WARNING)
# get the the logger with the name 'PIL'
pil_logger = logging.getLogger("PIL")
# override the logger logging level to INFO
pil_logger.setLevel(logging.INFO)

# flake8: noqa
log = logging.getLogger(__name__)

PLOT_IMGS_DIR = os.environ.get("ICON4PY_OUTPUT_DIR", "undefined_output_runxxx")
PLOT_FREQUENCY = int(os.environ.get("ICON4PY_PLOT_FREQUENCY", 1500))


def pickle_data(state, label: str = "") -> None:
    if not hasattr(pickle_data, "counter"):
        pickle_data.counter = 0
    else:
        pickle_data.counter += 1

    if type(state) is dict:
        state_dict = state
    else:
        state_dict = {
            "vn": state.vn.asnumpy(),
            "w": state.w.asnumpy(),
            "rho": state.rho.asnumpy(),
            "exner": state.exner.asnumpy(),
            "theta_v": state.theta_v.asnumpy(),
        }

    if not os.path.isdir(PLOT_IMGS_DIR):
        os.makedirs(PLOT_IMGS_DIR)
    if "debug_" in label:
        file_name = f"{PLOT_IMGS_DIR}/{pickle_data.counter:06d}_{label}.pkl"
    else:
        file_name = f"{PLOT_IMGS_DIR}/{label}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(state_dict, f)
        log.info(f"PLOTS: saved {file_name}")

