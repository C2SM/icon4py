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

from icon4py.model.atmosphere.dycore.stencils.compute_edge_diagnostics_for_dycore_and_update_vn import (
    compute_rho_theta_pgrad_and_update_vn,
)
from icon4py.model.common import model_backends, model_options
from icon4py.model.common.grid import simple


class _GraphCaptured(Exception):
    pass


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled, expected_maps", [(False, 6), (True, 5)])
def test_theta_fusion_reaches_optimizer(enabled: bool, expected_maps: int) -> None:
    """Check the model-to-compiler wiring without a GPU or native compilation."""
    maps: list[dace.nodes.MapEntry] = []

    def capture_graph(sdfg: dace.SDFG) -> None:
        maps.extend(
            node
            for state in sdfg.states()
            for node in state.nodes()
            if isinstance(node, dace.nodes.MapEntry) and state.entry_node(node) is None
        )
        # Stop after dataflow optimization, before generating a native library.
        raise _GraphCaptured

    grid = simple.simple_grid(num_levels=120)
    # Use a small mesh with distinct boundary/interior domains and the measured
    # 120-level vertical partition. The enabled arm uses normal model options.
    static_args = dict(
        limited_area=True,
        is_iau_active=False,
        nflatlev=27,
        nflat_gradp=119,
        start_edge_lateral_boundary=0,
        start_edge_lateral_boundary_level_7=3,
        start_edge_nudging_level_2=5,
        end_edge_nudging=5,
        end_edge_local=grid.num_edges,
        end_edge_halo=grid.num_edges,
        horizontal_start=0,
        horizontal_end=grid.num_edges,
        vertical_start=0,
        vertical_end=grid.num_levels,
    )
    with pytest.raises(_GraphCaptured):
        model_options.setup_program(
            program=compute_rho_theta_pgrad_and_update_vn,
            backend={
                "device": model_backends.CPU,
                "use_metrics": False,
                "optimization_args": {
                    "vertical_split_allow_shared_data": enabled,
                    "optimization_hooks": {
                        gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPost: capture_graph
                    },
                },
            },
            constant_args={
                name: value if isinstance(value, bool) else gtx.int32(value)
                for name, value in static_args.items()
            },
            offset_provider=grid.connectivities,
        )
    assert len(maps) == expected_maps
