"""The fused reconstruction+flux kernel against the two-stage form it replaces.

This is the half of Andreas Jocksch's 'upwind_hflux_miura_cell' reformulation that carries
over to GT4Py: fusing the least-squares reconstruction into the flux kernel so the three
coefficient fields are never written to memory. Whether it is worth having is an empirical
question, so this file both proves the two forms agree bit for bit and benchmarks them
against each other.
"""

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_from_linear_coefficients_alt import (
    compute_horizontal_tracer_flux_from_linear_coefficients_alt,
)
from icon4py.model.atmosphere.tracer_advection.stencils.reconstruct_and_compute_horizontal_tracer_flux_linear import (
    reconstruct_and_compute_horizontal_tracer_flux_linear,
)
from icon4py.model.atmosphere.tracer_advection.stencils.reconstruct_linear_coefficients_svd import (
    reconstruct_linear_coefficients_svd,
)
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc


#: Enough levels that the kernel is bandwidth bound rather than launch bound, which is the
#: regime the whole argument for fusing is about. This is between the level count of the
#: ordinary grid presets (40) and of the benchmark ones (80), so the benchmark runs on the
#: latter and skips on the former.
_BENCHMARK_NUM_LEVELS = 64


def _inputs(grid: base.Grid, backend: gtx_typing.Backend | None) -> dict:
    allocator = model_backends.get_allocator(backend)
    return {
        "p_cc": data_alloc.random_field(grid, dims.CellDim, dims.KDim, allocator=allocator),
        "lsq_pseudoinv_1": data_alloc.random_field(
            grid, dims.CellDim, dims.C2E2CDim, allocator=allocator
        ),
        "lsq_pseudoinv_2": data_alloc.random_field(
            grid, dims.CellDim, dims.C2E2CDim, allocator=allocator
        ),
        "distv_bary_1": data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        "distv_bary_2": data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        "p_mass_flx_e": data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        # a sign-mixed normal velocity, so both branches of the upwind select are taken
        "p_vn": data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, low=-1.0, high=1.0, allocator=allocator
        ),
    }


def _domain(grid: base.Grid) -> dict:
    edge_domain = h_grid.domain(dims.EdgeDim)
    cell_domain = h_grid.domain(dims.CellDim)
    return {
        "edge": {
            "horizontal_start": grid.start_index(
                edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
            ),
            "horizontal_end": gtx.int32(grid.num_edges),
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(grid.num_levels),
        },
        "cell": {
            "horizontal_start": grid.start_index(
                cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
            ),
            "horizontal_end": gtx.int32(grid.num_cells),
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(grid.num_levels),
        },
    }


def _run_two_stage(
    grid: base.Grid, backend: gtx_typing.Backend | None, inputs: dict, buffers: dict
) -> gtx.Field:
    domain = _domain(grid)
    reconstruct_linear_coefficients_svd.with_backend(backend)(
        p_cc=inputs["p_cc"],
        lsq_pseudoinv_1=inputs["lsq_pseudoinv_1"],
        lsq_pseudoinv_2=inputs["lsq_pseudoinv_2"],
        p_coeff_1_dsl=buffers["p_coeff_1"],
        p_coeff_2_dsl=buffers["p_coeff_2"],
        p_coeff_3_dsl=buffers["p_coeff_3"],
        offset_provider=grid.connectivities,
        **domain["cell"],
    )
    compute_horizontal_tracer_flux_from_linear_coefficients_alt.with_backend(backend)(
        z_lsq_coeff_1=buffers["p_coeff_1"],
        z_lsq_coeff_2=buffers["p_coeff_2"],
        z_lsq_coeff_3=buffers["p_coeff_3"],
        distv_bary_1=inputs["distv_bary_1"],
        distv_bary_2=inputs["distv_bary_2"],
        p_mass_flx_e=inputs["p_mass_flx_e"],
        p_vn=inputs["p_vn"],
        p_out_e=buffers["p_out_e_two_stage"],
        offset_provider=grid.connectivities,
        **domain["edge"],
    )
    return buffers["p_out_e_two_stage"]


def _run_fused(
    grid: base.Grid, backend: gtx_typing.Backend | None, inputs: dict, buffers: dict
) -> gtx.Field:
    reconstruct_and_compute_horizontal_tracer_flux_linear.with_backend(backend)(
        **inputs,
        p_out_e=buffers["p_out_e_fused"],
        offset_provider=grid.connectivities,
        **_domain(grid)["edge"],
    )
    return buffers["p_out_e_fused"]


def _buffers(grid: base.Grid, backend: gtx_typing.Backend | None) -> dict:
    allocator = model_backends.get_allocator(backend)
    return {
        name: data_alloc.zero_field(grid, *field_dims, allocator=allocator)
        for name, field_dims in (
            ("p_coeff_1", (dims.CellDim, dims.KDim)),
            ("p_coeff_2", (dims.CellDim, dims.KDim)),
            ("p_coeff_3", (dims.CellDim, dims.KDim)),
            ("p_out_e_two_stage", (dims.EdgeDim, dims.KDim)),
            ("p_out_e_fused", (dims.EdgeDim, dims.KDim)),
        )
    }


def test_fused_flux_matches_the_two_stage_form(
    grid: base.Grid, backend: gtx_typing.Backend
) -> None:
    """Bit-identical, not merely close: the fused form must be a pure schedule change."""
    inputs = _inputs(grid, backend)
    buffers = _buffers(grid, backend)

    two_stage = _run_two_stage(grid, backend, inputs, buffers).asnumpy()
    fused = _run_fused(grid, backend, inputs, buffers).asnumpy()

    edge_start = _domain(grid)["edge"]["horizontal_start"]
    np.testing.assert_array_equal(fused[edge_start:], two_stage[edge_start:])


@pytest.mark.parametrize("formulation", ["two_stage", "fused"])
def test_benchmark_fused_against_two_stage(
    benchmark: pytest.FixtureRequest,
    formulation: str,
    grid: base.Grid,
    backend: gtx_typing.Backend,
) -> None:
    """Time the two formulations against each other; compare the two reported means.

    Needs a performance backend and a grid big enough for the comparison to mean anything:

        pytest --backend gtfn_cpu --grid icon_benchmark_regional \
            -k test_benchmark_fused_against_two_stage
    """
    if grid.num_levels < _BENCHMARK_NUM_LEVELS:
        pytest.skip(
            f"the comparison is only meaningful when bandwidth bound, so it wants at least "
            f"{_BENCHMARK_NUM_LEVELS} levels, but this grid has {grid.num_levels}; run it "
            f"with '--grid icon_benchmark_regional'"
        )
    inputs = _inputs(grid, backend)
    buffers = _buffers(grid, backend)
    run = _run_two_stage if formulation == "two_stage" else _run_fused

    # warm the compile cache outside the timed region
    run(grid, backend, inputs, buffers)
    benchmark(run, grid, backend, inputs, buffers)
