# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config
from icon4py.model.common import constants
from icon4py.model.testing import definitions as test_defs
from icon4py.model.testing.fixtures.datatest import (
    download_ser_data,
    experiment,
    experiment_description,
    process_props,
)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.EXCLAIM_APE_AES])
def test_experiment_config_has_the_tmx_surface(experiment: test_defs.Experiment) -> None:
    # the archive's nh_testcase_nml: isrfc_type = 1, ape_sst_val = 30, shflx and lhflx unset
    assert experiment.config.tmx is not None
    assert experiment.config.tmx_surface == tmx_config.TmxSurfaceConfig(
        surface_flux_type=tmx_config.SurfaceFluxType(1),
        kinematic_sensible_heat_flux=0.1,
        kinematic_latent_heat_flux=0.0,
        sea_surface_temperature=constants.MELTING_TEMPERATURE + 30.0,
    )
