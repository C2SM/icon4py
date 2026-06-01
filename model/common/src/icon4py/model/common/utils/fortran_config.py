# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib
from typing import Final, TypeVar

from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs


_T = TypeVar("_T")

NAMELIST_ATM_FNAME: Final = "NAMELIST_ICON_output_atm"
NAMELIST_MASTER_FNAME: Final = "icon_master.namelist"

ATM_DICT_FNAME: Final = f"{NAMELIST_ATM_FNAME}.json"
MASTER_DICT_FNAME: Final = f"{NAMELIST_MASTER_FNAME}.json"

def get_atm_dict_path(
    experiment_description: test_defs.ExperimentDescription,
    processor_props: test_defs.ProcessProperties,
) -> pathlib.Path:
    experiment_dir = dt_utils.get_ranked_experiment_name_with_version(
        experiment_description,
        processor_props.comm_size,
    )
    return experiment_dir / f"{NAMELIST_ATM_FNAME}.json"

def get_master_dict_path(
    experiment_description: test_defs.ExperimentDescription,
    processor_props: test_defs.ProcessProperties,
) -> pathlib.Path:
    experiment_dir = dt_utils.get_ranked_experiment_name_with_version(
        experiment_description,
        processor_props.comm_size,
    )
    return experiment_dir / f"{NAMELIST_MASTER_FNAME}.json"


def list_to_value(obj: list[_T] | _T) -> _T:
    # Some parameters are allocated as `max_dom`-sized lists, with one value
    # per domain. ICON4Py (for now) only runs on one domain.
    # Most parameters have the same value for all elements, others (such as
    # num_levels) have a default value different from domain[0].
    # Tracers are an even different case where there is one value per tracer,
    # but with the current version of ICON4Py all tracers get the same config.
    return obj[0] if isinstance(obj, list) else obj
