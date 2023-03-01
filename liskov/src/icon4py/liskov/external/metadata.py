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
import subprocess

import click
from typing_extensions import Any

from icon4py.liskov.external.exceptions import (
    MissingClickContextError,
    MissingGitError,
)


class CodeMetadata:
    """Class that handles retrieval of icon-liskov runtime metadata."""

    def __init__(self) -> None:
        self.generated_on = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    def commit_hash(self) -> str:
        """Get the latest git commit hash."""
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            )
        except Exception as e:
            raise MissingGitError(
                f"Git is not available or there is no commit or tag.\n {e}"
            )

    @property
    def tag(self) -> str:
        """Get the latest git tag."""
        try:
            return (
                subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
                .decode()
                .strip()
            )
        except Exception as e:
            raise MissingGitError(
                f"Git is not available or there is no commit or tag.\n {e}"
            )
