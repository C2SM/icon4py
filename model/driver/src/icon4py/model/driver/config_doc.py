# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from rich.progress import Progress, SpinnerColumn, TextColumn


def main() -> None:
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as process:
        process.add_task(description="Loading ICON4Py Datatypes ...", total=None)
        # these non-top-level imports are necessary as long as they take
        # more than about half a second combined
        from icon4py.model.common.config.config_doc import ConfigDocApp  # noqa: PLC0415
        from icon4py.model.driver.config import ExperimentConfig  # noqa: PLC0415
    app = ConfigDocApp(root_class=ExperimentConfig)
    app.run()
