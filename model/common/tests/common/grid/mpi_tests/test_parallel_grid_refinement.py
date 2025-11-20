import pytest
try:
    import mpi4py

    import mpi4py.MPI
except ImportError:
    pytest.skip("Skipping parallel on single node installation", allow_module_level=True)

import numpy as np
import gt4py.next as gtx
from icon4py.model.testing import definitions, serialbox
from icon4py.model.common.decomposition import definitions as decomposition, mpi_decomposition
from icon4py.model.common.grid import grid_refinement, horizontal as h_grid
from icon4py.model.common import dimension as dims
from .. import utils
from ..fixtures import backend, experiment, grid_savepoint, icon_grid, data_provider, download_ser_data, ranked_data_path, processor_props
@pytest.mark.parametrize("processor_props", [True], indirect=True)
#@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
@pytest.mark.parametrize("dim", (dims.EdgeDim,))
@pytest.mark.parametrize("halos", (h_grid.Zone.HALO,))
@pytest.mark.parametrize("experiment", (definitions.Experiments.MCH_CH_R04B09,))
@pytest.mark.mpi
def test_halo_start_end_index(dim: gtx.Dimension, halos: h_grid.Zone, experiment:definitions.Experiment, grid_savepoint: serialbox.IconGridSavepoint,
                              processor_props:decomposition.ProcessProperties)->None:
    ref_grid = grid_savepoint.construct_icon_grid(None, keep_skip_values=True)
    print(f"{processor_props.rank}/{processor_props.comm_size} - ref start {grid_savepoint.edge_start_index()}")
    print(f"{processor_props.rank}/{processor_props.comm_size} - ref start {grid_savepoint.edge_end_index()}")

    decomposition_info = grid_savepoint.construct_decomposition_info()
    refin_ctrl = {dim: grid_savepoint.refin_ctrl(dim) for dim in utils.main_horizontal_dims()}
    domain = h_grid.domain(dim)(halos)
    print(f"rank = {processor_props.rank}/{processor_props.comm_size} - refinement ctrl {refin_ctrl[dim]}")
    processor_props.comm.Barrier()

    start_indices, end_indices = grid_refinement.compute_domain_bounds(dim, refin_ctrl, decomposition_info, processor_props.rank)
    print(f"rank = {processor_props.rank}/{processor_props.comm_size} - start {start_indices}")
    print(f"rank = {processor_props.rank}/{processor_props.comm_size} - end {end_indices}")
    ref_start_index = ref_grid.start_index(domain)
    ref_end_index = ref_grid.end_index(domain)
    computed_start = start_indices[domain]
    computed_end = end_indices[domain]
    assert computed_start == ref_start_index, f"{processor_props.rank}/{processor_props.comm_size} - experiment = {experiment.name}: start_index for {domain} does not match: is {computed_start}, expected {ref_start_index}"
    assert computed_end == ref_end_index, f"{processor_props.rank}/{processor_props.comm_size} - experiment = {experiment.name}: end_index for {domain} does not match: is {computed_end}, expected {ref_end_index}"


