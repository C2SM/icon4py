# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Which ICON build produced the serialized data a test session is using.

ICON prints its 'git describe' at startup and the generation script copies that log
into every archive, so the provenance of every archive ever published is already on
disk. Reading it back turns 'these reference values disagree with the code' into
'these reference values came from ICON <revision>', which is the difference between
hours of archaeology and one 'git log'.

Deliberately free of heavy imports: the pytest plugin that renders this is registered
for every session in the repository, including ones with no serialized data at all.
"""

from __future__ import annotations

import pathlib
import re


_REVISION = re.compile(r"^ revision: (?P<revision>\S+)$", re.MULTILINE)

# Archive directory name -> ICON revision, for the session's terminal summary.
_seen: dict[str, str] = {}


def read_icon_revision(archive_dir: pathlib.Path) -> str | None:
    """The 'git describe' of the ICON build that wrote an archive, if it says.

    The full describe string is returned rather than the bare commit: it is a valid
    git revision, so 'git log OLD..NEW' works on it directly, and its release lineage
    distinguishes a two-commit bump from a jump to another ICON release.
    """
    for log in sorted(archive_dir.glob("LOG.*.o")):
        match = _REVISION.search(log.read_text(errors="replace"))
        if match:
            return match["revision"]
    return None


def record(archive_name: str, archive_dir: pathlib.Path) -> None:
    """Note the provenance of an archive this session uses. Never raises."""
    if archive_name in _seen:
        return
    try:
        _seen[archive_name] = read_icon_revision(archive_dir) or "unknown"
    except OSError:
        _seen[archive_name] = "unknown"


def seen() -> dict[str, str]:
    return dict(_seen)


def merge(recorded: dict[str, str]) -> None:
    """Take in what another process saw, so that xdist workers reach the summary."""
    _seen.update(recorded)
