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

from pathlib import Path

from icon4py.decomposition.parallel_setup import get_processor_properties


base_path = Path(__file__).parent.parent.parent.parent.joinpath("testdata/ser_icondata")

props = get_processor_properties(with_mpi=True)
path = base_path.joinpath(f"mpitask{props.comm_size}/mch_ch_r04b09_dsl/ser_data")
