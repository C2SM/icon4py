# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common.config.config_doc import ConfigDocApp
from icon4py.model.driver.config import ExperimentConfig


def main() -> None:
    app = ConfigDocApp(root_class=ExperimentConfig)
    app.run()
