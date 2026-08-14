# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reading the ICON revision out of an archive."""

from __future__ import annotations

import pathlib

from icon4py.model.testing import provenance


REVISION = "icon-2026.04-dwd-1.9-136-g1df503335b8726dc445659c21020f4f09d9acc3d"

# ICON's startup banner as it appears in the slurm log, with the surrounding output that
# makes the line have to be searched for, and the nested component that must not match.
LOG = f""" executable: /capstor/store/.../build_serialize/bin/icon
 version: 2026.04
 revision: {REVISION}
 repository: git@gitlab.dkrz.de:icon/icon-nwp.git
 model components:
   ICON-Land:
     revision: icon-land-2026.04-18-geb24d7431a16ef66bae693a7ef95c32ca47862ff
 master_control: start model initialization.
"""


def test_read_icon_revision(tmp_path: pathlib.Path) -> None:
    (tmp_path / "LOG.exp.foo_sb.run.986945.o").write_text(LOG)

    assert provenance.read_icon_revision(tmp_path) == REVISION
