# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
from typing import Any

import click

from icon4pytools import __version__
from icon4pytools.liskov.external.exceptions import MissingClickContextError


class CodeMetadata:
    """Class that handles retrieval of icon-liskov runtime metadata."""

    @property
    def generated_on(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @property
    def cli_params(self) -> dict[str, Any]:
        try:
            ctx = click.get_current_context()
            if ctx is None:
                raise MissingClickContextError("No active Click context found.")

            params = ctx.params.copy()

            if ctx.parent is not None:
                params.update(ctx.parent.params)

            return params
        except Exception as err:
            raise MissingClickContextError(
                f"Cannot fetch click context in this thread as no click command has been executed.\n {err}"
            ) from err

    @property
    def version(self) -> str:
        """Get the current version."""
        return __version__
