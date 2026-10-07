# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from importlib.resources import files

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn


tapp = typer.Typer()


@tapp.command()
def main(try_cache: bool = True, regen_cache: bool = True) -> None:
    cache_file = files("icon4py.model.driver").joinpath("config_doc_cache.yaml")
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        # these non-top-level imports are necessary as long as they take
        # more than about half a second combined
        progress.add_task(description="", total=None)
        from icon4py.model.common.config.config_doc import ConfigDocApp  # noqa: PLC0415

        if cache_file.is_file() and try_cache:
            progress.add_task(description="Loading from cache ...", total=None)
            app = ConfigDocApp(
                root_class=None, try_cache=try_cache, regen_cache=regen_cache, cache_path=cache_file
            )
        else:
            progress.add_task(description="Loading ICON4Py Datatypes ...", total=None)
            from icon4py.model.driver.config import ExperimentConfig  # noqa: PLC0415

            app = ConfigDocApp(
                root_class=ExperimentConfig,
                try_cache=try_cache,
                regen_cache=regen_cache,
                cache_path=cache_file,
            )
    app.run()
