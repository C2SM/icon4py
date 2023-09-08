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
import wget
from pathlib import Path
import pytest
from gt4py.next.program_processors.runners.roundtrip import executor
from gt4py.next.program_processors.runners import gtfn_cpu

from icon4py.model.common.test_utils.simple_mesh import SimpleMesh
from icon4py.model.common.test_utils.serialbox_utils import IconSerialDataProvider


data_uri = "https://polybox.ethz.ch/index.php/s/LcAbscZqnsx4WCf/download"
data_path = Path(__file__).parent.joinpath("ser_icondata")
extracted_path = data_path.joinpath("mch_ch_r04b09_dsl/ser_data")
data_file = data_path.joinpath("mch_ch_r04b09_dsl_v2.tar.gz").name


BACKENDS = {
    #"embedded": executor,
    "gtfn": gtfn_cpu.run_gtfn_cached,
    "gtfn_temporaries": gtfn_cpu.run_gtfn_with_temporaries_cached
}
MESHES = {
    "simple_mesh": SimpleMesh()
}


@pytest.fixture
def get_data_path(setup_icon_data):
    return extracted_path


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
def icon_grid(grid_savepoint):
    """
    Load the icon grid from an ICON savepoint.

    Uses the special grid_savepoint that contains data from p_patch
    """
    return grid_savepoint.construct_icon_grid()


@pytest.fixture
def mesh(icon_grid):
    return icon_grid

@pytest.fixture
def grid_savepoint(data_provider):
    return data_provider.from_savepoint_grid()


#@pytest.fixture(
#    ids=MESHES.keys(),
#    params=MESHES.values(),
#)
#def mesh(request):
#    return request.param


@pytest.fixture(ids=BACKENDS.keys(), params=BACKENDS.values())
def backend(request):
    return request.param
