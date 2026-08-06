# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Which ICON build produced the serialized data a test session is using."""

from __future__ import annotations

import pathlib
import re


_REVISION = re.compile(r"^ revision: (\S+)$")

# Archive directory name -> ICON revision, rendered in the terminal summary.
seen: dict[str, str] = {}


def read_icon_revision(archive_dir: pathlib.Path) -> str | None:
    """The 'git describe' of the ICON build that wrote an archive, if it says.

    The full describe string rather than the bare commit: it is a valid git revision,
    so 'git log OLD..NEW' works on it directly.
    """
    for log in sorted(archive_dir.glob("LOG.*.o")):
        with log.open(errors="replace") as lines:
            for line in lines:
                match = _REVISION.match(line.rstrip("\n"))
                if match:
                    return match[1]
    return None


def record(archive_name: str, archive_dir: pathlib.Path) -> None:
    """Note the provenance of an archive this session uses. Never raises."""
    if archive_name in seen:
        return
    try:
        seen[archive_name] = read_icon_revision(archive_dir) or "unknown"
    except OSError:
        seen[archive_name] = "unknown"
