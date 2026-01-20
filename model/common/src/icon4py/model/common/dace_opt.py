# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace
from dace.transformation import dataflow as dace_dataflow


def graupel_run_top_level_post(sdfg: dace.SDFG) -> None:
    sdfg.apply_transformations_repeated(dace_dataflow.TaskletFusion, validate=False)
    sdfg.save("graupel_run_top_level_post.sdfg")
