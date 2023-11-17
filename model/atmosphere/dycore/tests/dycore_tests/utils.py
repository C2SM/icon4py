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

from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import NonHydrostaticConfig


def mch_ch_r04b09_dsl_nonhydrostatic_config(ndyn_substeps):
    """Create configuration matching the mch_chR04b09_dsl experiment."""
    config = NonHydrostaticConfig(
        ndyn_substeps_var=ndyn_substeps,
        divdamp_order=24,
        iau_wgt_dyn=1.0,
        divdamp_fac=0.004,
        max_nudging_coeff=0.075,
    )
    return config


def exclaim_ape_nonhydrostatic_config(ndyn_substeps):
    """Create configuration for EXCLAIM APE experiment."""
    return NonHydrostaticConfig(
        rayleigh_coeff=0.1,
        divdamp_order=24,
        ndyn_substeps_var=ndyn_substeps,
        ltestcase=True,
    )


def construct_config(name: str, ndyn_substeps: int = 5):
    if name.lower() in "mch_ch_r04b09_dsl":
        return mch_ch_r04b09_dsl_nonhydrostatic_config(ndyn_substeps)
    elif name.lower() in "exclaim_ape_r02b04":
        return exclaim_ape_nonhydrostatic_config(ndyn_substeps)
