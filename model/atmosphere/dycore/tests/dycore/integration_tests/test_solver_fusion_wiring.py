# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace  # type: ignore[import-untyped]
import gt4py.next as gtx
import pytest
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from icon4py.model.atmosphere.dycore.stencils import vertically_implicit_dycore_solver as solvers
from icon4py.model.common import model_backends, model_options
from icon4py.model.common.grid import simple


class _GraphCaptured(Exception):
    pass


@pytest.mark.cpu_only
@pytest.mark.parametrize("stage", ["predictor", "corrector"])
@pytest.mark.parametrize("enabled", [False, True])
def test_solver_coefficients_reach_scan(stage: str, enabled: bool) -> None:
    """Catch a skipped fusion even when its option and numerical results are valid."""
    coefficient_scopes: list[dace.SDFG] = []

    def capture_graph(sdfg: dace.SDFG) -> None:
        nested_graphs = list(sdfg.all_sdfgs_recursive())
        assert any(
            isinstance(node, dace.sdfg.state.LoopRegion)
            for graph in nested_graphs
            for node in graph.nodes()
        )
        # The forward recurrence takes a, b, c and d before their producers
        # are fused. Afterwards it reads the underlying model fields instead.
        coefficient_scopes.extend(
            graph for graph in nested_graphs if {"a", "b", "c", "d"} <= graph.arrays.keys()
        )
        raise _GraphCaptured

    grid = simple.simple_grid(num_levels=120)
    static_args = dict(
        is_iau_active=False,
        rayleigh_type=2,
        at_first_substep=False,
        end_index_of_damping_layer=1,
        kstart_moist=0,
        start_cell_index_nudging=3,
        end_cell_index_local=grid.num_cells,
        vertical_start_index_model_top=0,
        vertical_end_index_model_surface=grid.num_levels + 1,
    )
    if stage == "predictor":
        static_args.update(
            divdamp_type=3,
            flat_level_index_plus1=27,
            start_cell_index_lateral_lvl3=0,
            end_cell_index_halo_lvl1=grid.num_cells,
        )
    else:
        static_args.update(at_last_substep=False, prepare_fluxes_for_advection=True)
    with pytest.raises(_GraphCaptured):
        model_options.setup_program(
            program=getattr(solvers, f"vertically_implicit_solver_at_{stage}_step"),
            backend={
                "device": model_backends.CPU,
                "use_metrics": False,
                "optimization_args": {
                    "fuse_scan_inputs": enabled,
                    "optimization_hooks": {
                        gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPre: capture_graph
                    },
                },
            },
            constant_args={
                name: value if isinstance(value, bool) else gtx.int32(value)
                for name, value in static_args.items()
            },
            offset_provider=grid.connectivities,
        )
    assert len(coefficient_scopes) == (0 if enabled else 1)
