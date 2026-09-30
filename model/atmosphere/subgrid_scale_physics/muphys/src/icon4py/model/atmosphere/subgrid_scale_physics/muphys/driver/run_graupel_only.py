#!/usr/bin/env python
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

from gt4py import next as gtx
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import utils
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations import (
    graupel,
    graupel_dace_hooks,
)
from icon4py.model.common import model_backends, model_options, type_alias as ta


def setup_graupel(
    *,
    dt: float,
    qnc: float,
    backend: model_backends.BackendLike,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
    enable_masking: bool = True,
    enable_dace_hooks: bool = False,
):
    if enable_dace_hooks and model_backends.is_backend_descriptor(backend):
        # The graupel scan needs two dace auto-opt hooks. They can only be injected into
        # the backend *descriptor* (a dict), before it is turned into a concrete backend.
        # A concrete/resolved backend (e.g. the gtfn backend the standalone driver threads
        # in) is not a descriptor: gtfn does not use these hooks, so we pass it through
        # unmodified. The dace path always drives muphys with a descriptor, so it still
        # gets the hooks.
        backend = copy.deepcopy(backend)
        if "optimization_args" not in backend:
            backend["optimization_args"] = {}
        backend["optimization_args"]["optimization_hooks"] = {
            gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPre: graupel_dace_hooks.remove_self_copy_inside_scan,
            gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPost: graupel_dace_hooks.rename_intermediate_access_nodes,
        }
    with utils.recursion_limit(10**4):  # TODO(havogt): make an option in gt4py?
        graupel_run_program = model_options.setup_program(
            backend=backend,
            program=graupel.graupel_run,
            constant_args={
                "dt": ta.wpfloat(dt),
                "qnc": ta.wpfloat(qnc),
                "enable_masking": enable_masking,
            },
            horizontal_sizes={
                "horizontal_start": horizontal_start,
                "horizontal_end": horizontal_end,
            },
            vertical_sizes={
                "vertical_start": vertical_start,
                "vertical_end": vertical_end,
            },
            offset_provider={},
        )
        gtx.wait_for_compilation()
        return graupel_run_program
