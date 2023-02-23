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
import tarfile
from pathlib import Path

import numpy as np
import pytest
import wget

from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2VDim,
    E2CDim,
    E2VDim,
    EdgeDim,
    V2EDim,
    VertexDim,
)
from icon4py.diffusion.diffusion import DiffusionConfig
from icon4py.diffusion.horizontal import HorizontalMeshSize
from icon4py.diffusion.icon_grid import IconGrid, MeshConfig, VerticalMeshConfig
from icon4py.testutils.serialbox_utils import IconSerialDataProvider


data_uri = "https://polybox.ethz.ch/index.php/s/rzuvPf7p9sM801I/download"
data_path = Path(__file__).parent.joinpath("ser_icondata")
extracted_path = data_path.joinpath("mch_ch_r04b09_dsl/ser_data")
data_file = data_path.joinpath("mch_ch_r04b09_dsl_v2.tar.gz").name


@pytest.fixture(scope="session")
def setup_icon_data():
    """
    Get the binary ICON data from a remote server.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    data_path.mkdir(parents=True, exist_ok=True)
    if not any(data_path.iterdir()):
        print(
            f"directory {data_path} is empty: downloading data from {data_uri} and extracting"
        )
        wget.download(data_uri, out=data_file)
        # extract downloaded file
        if not tarfile.is_tarfile(data_file):
            raise NotImplementedError(f"{data_file} needs to be a valid tar file")
        with tarfile.open(data_file, mode="r:*") as tf:
            tf.extractall(path=data_path)
        Path(data_file).unlink(missing_ok=True)


@pytest.fixture
def data_provider(setup_icon_data) -> IconSerialDataProvider:
    return IconSerialDataProvider("icon_pydycore", str(extracted_path), True)


@pytest.fixture
def linit():
    """
    Set the 'linit' flag for the ICON diffusion data savepoint.

    Defaults to False
    """
    return False


@pytest.fixture
def step_date_init():
    """
    Set the step date for the loaded ICON time stamp at start of module.

    Defaults to 2021-06-20T12:00:10.000'
    """
    return "2021-06-20T12:00:10.000"


@pytest.fixture
def step_date_exit():
    """
    Set the step date for the loaded ICON time stamp at the end of module.

    Defaults to 2021-06-20T12:00:10.000'
    """
    return "2021-06-20T12:00:10.000"


@pytest.fixture
def diffusion_savepoint_init(data_provider, linit, step_date_init):
    """
    Load data from ICON savepoint at start of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'

    linit flag can be set by overriding the 'linit' fixture
    """
    return data_provider.from_savepoint_diffusion_init(linit=linit, date=step_date_init)


@pytest.fixture
def diffusion_savepoint_exit(data_provider, step_date_exit):
    """
    Load data from ICON savepoint at exist of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    sp = data_provider.from_savepoint_diffusion_exit(linit=False, date=step_date_exit)
    return sp


@pytest.fixture
def icon_grid(data_provider):
    """
    Load the icon grid from an ICON savepoint.

    Uses the default save_point from 'savepoint_init' fixture, however these data don't change for
    different time steps.
    """
    sp = data_provider.from_savepoint_grid()
    sp_meta = sp.get_metadata("nproma", "nlev", "num_vert", "num_cells", "num_edges")

    cell_starts = sp.cells_start_index()
    cell_ends = sp.cells_end_index()
    vertex_starts = sp.vertex_start_index()
    vertex_ends = sp.vertex_end_index()
    edge_starts = sp.edge_start_index()
    edge_ends = sp.edge_end_index()

    config = MeshConfig(
        HorizontalMeshSize(
            num_vertices=sp_meta["nproma"],  # or rather "num_vert"
            num_cells=sp_meta["nproma"],  # or rather "num_cells"
            num_edges=sp_meta["nproma"],  # or rather "num_edges"
        ),
        VerticalMeshConfig(num_lev=sp_meta["nlev"]),
    )

    c2e2c = sp.c2e2c()
    c2e2c0 = np.column_stack((c2e2c, (np.asarray(range(c2e2c.shape[0])))))
    grid = (
        IconGrid()
        .with_config(config)
        .with_start_end_indices(VertexDim, vertex_starts, vertex_ends)
        .with_start_end_indices(EdgeDim, edge_starts, edge_ends)
        .with_start_end_indices(CellDim, cell_starts, cell_ends)
        .with_connectivities(
            {C2EDim: sp.c2e(), E2CDim: sp.e2c(), C2E2CDim: c2e2c, C2E2CODim: c2e2c0}
        )
        .with_connectivities({E2VDim: sp.e2v(), V2EDim: sp.v2e(), E2C2VDim: sp.e2c2v()})
    )
    return grid


@pytest.fixture
def grid_savepoint(data_provider):
    return data_provider.from_savepoint_grid()


@pytest.fixture
def r04b09_diffusion_config(setup_icon_data) -> DiffusionConfig:
    """
    Create DiffusionConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return DiffusionConfig(
        diffusion_type=5,
        hdiff_w=True,
        hdiff_vn=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=True,
        velocity_boundary_diffusion_denom=150.0,
        max_nudging_coeff=0.075,
    )


@pytest.fixture
def damping_height():
    return 12500
