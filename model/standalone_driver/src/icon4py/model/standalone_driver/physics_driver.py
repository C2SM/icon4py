# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import config as driver_config


log = logging.getLogger(__name__)


class PhysicsDriver:
    def __init__(
        self,
        config: driver_config.DriverConfig,
    ):
        self.config = config

    def _local_fields(self, grid: IconGrid):
        saved_exner = data_alloc.allocate_cell_field(
            grid,
            dims.EdgeDim,
            dims.KDim,
            dtype=ta.wpfloat,
            name="saved_exner",
        )

    def __call__(self, *args, **kwds):
        diagnose_uv()  # 1: min_rlcell_int
        diagnose_virtual_temperature_and_temperature()  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        data_alloc.field(saved_exner)  # , 1: min_rlcell
        saturation_adjusment()  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        diagnose_exner_and_virtual_temperature()  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        diagnose_pressure()  # from kmoist, grf_bdywidth_c+1: min_rlcell_int
        simple_surface()  # only for qv_s, grf_bdywidth_c+1: min_rlcell_int
        # turbulence() # grf_bdywidth_c+1: min_rlcell_int, NOT PORTED
        microphysics()
        saturation_adjustment()
        # diagnose_virtual_temperature_and_exner_and_theta_v() # grf_bdywidth_c+1: min_rlcell_int
        # diagnose_pressure() # grf_bdywidth_c+1: min_rlcell_int
        # surface_transfer() # grf_bdywidth_c+1: min_rlcell_int, NOT PORTED
        # halo exchange ddt_u_turb and ddt_v_turb
        diagnose_exner_and_theta_v()  # min_rlcell_int-1: min_rlcell_int
        update_vn_from_turb()  # grf_bdywidth_e+1: min_rledge_int
