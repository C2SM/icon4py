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
import os
import tarfile

import numpy as np
import pytest
import wget

from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CellDim,
    E2CDim,
    E2VDim,
    EdgeDim,
    V2EDim,
    VertexDim,
)
from icon4py.diffusion.diffusion import DiffusionConfig
from icon4py.diffusion.horizontal import HorizontalMeshConfig
from icon4py.diffusion.icon_grid import (
    IconGrid,
    MeshConfig,
    VerticalMeshConfig,
    VerticalModelParams,
)
from icon4py.testutils.serialbox_utils import IconSerialDataProvider


data_uri = "https://polybox.ethz.ch/index.php/s/kP0Q2dDU6DytEqI/download"
data_path = os.path.join(os.path.dirname(__file__), "./ser_icondata")
extracted_path = os.path.join(data_path, "mch_ch_r04b09_dsl/ser_data")
data_file = os.path.join(data_path, "ser_data_diffusion.tar.gz")


@pytest.fixture(scope="session")
def setup_icon_data():
    os.makedirs(data_path, exist_ok=True)
    if len(os.listdir(data_path)) == 0:
        print(
            f"directory {data_path} is empty: downloading data from {data_uri} and extracting"
        )

        wget.download(data_uri, out=data_file)
        # extract downloaded file
        if not tarfile.is_tarfile(data_file):
            raise NotImplementedError(f"{data_file} needs to be a valid tar file")
        with tarfile.open(data_file, mode="r:*") as tf:
            tf.extractall(path=data_path)
        os.remove(data_file)


@pytest.fixture
def linit():
    return False


@pytest.fixture
def step_date():
    return "2021-06-20T12:00:10.000"


@pytest.fixture
def savepoint_init(setup_icon_data, linit, step_date):
    sp = IconSerialDataProvider(
        "icon_diffusion_init", extracted_path, True
    ).from_savepoint_init(linit=linit, date=step_date)
    return sp


@pytest.fixture
def savepoint_exit(setup_icon_data, step_date):
    sp = IconSerialDataProvider(
        "icon_diffusion_exit", extracted_path, True
    ).from_savepoint_init(linit=False, date=step_date)
    return sp


@pytest.fixture
def icon_grid(savepoint_init):
    sp = savepoint_init

    sp_meta = sp.get_metadata("nproma", "nlev", "num_vert", "num_cells", "num_edges")

    cell_starts = sp.cells_start_index()
    cell_ends = sp.cells_end_index()
    vertex_starts = sp.vertex_start_index()
    vertex_ends = sp.vertex_end_index()
    edge_starts = sp.edge_start_index()
    edge_ends = sp.edge_end_index()

    config = MeshConfig(
        HorizontalMeshConfig(
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
        .with_connectivities({E2VDim: sp.e2v(), V2EDim: sp.v2e()})
    )
    return grid


@pytest.fixture
def r04b09_diffusion_config(setup_icon_data) -> DiffusionConfig:
    """
    Create DiffusionConfig matching MCH_CH_r04b09_dsl.

    Sets values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    sp = IconSerialDataProvider(
        "icon_diffusion_init", extracted_path, True
    ).from_savepoint_init(linit=True, date="2021-06-20T12:00:10.000")
    nproma = sp.get_metadata("nproma")["nproma"]
    num_lev = sp.get_metadata("nlev")["nlev"]
    horizontal_config = HorizontalMeshConfig(
        num_vertices=nproma, num_cells=nproma, num_edges=nproma
    )
    vertical_config = VerticalMeshConfig(num_lev=num_lev)

    grid = IconGrid().with_config(
        MeshConfig(horizontal_config=horizontal_config, vertical_config=vertical_config)
    )

    verticalParams = VerticalModelParams(
        rayleigh_damping_height=12500, vct_a=sp.vct_a()
    )

    return DiffusionConfig(
        grid=grid,
        vertical_params=verticalParams,
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
