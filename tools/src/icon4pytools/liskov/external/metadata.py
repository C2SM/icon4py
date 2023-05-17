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
import datetime
from typing import Any

import click
import icon4pytools
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
            return ctx.params
        except Exception as e:
            raise MissingClickContextError(
                f"Cannot fetch click context in this thread as no click command has been executed.\n {e}"
            )

    @property
    def version(self) -> str:
        """Get the current version."""
        return icon4pytools.__version__
