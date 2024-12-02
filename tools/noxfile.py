# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import nox


nox.options.default_venv_backend = "uv|virtualenv"


@nox.session
@nox.parametrize("python", ["3.10", "3.11"])
def tests(session):
    root_project = nox.project.load_toml("../pyproject.toml")
    dev_workspace_group = nox.project.dependency_groups(root_project, "test")
    session.install(*dev_workspace_group)
    session.install(".")
    num_processes = session.env.get("NUM_PROCESSES", "1")
    session.run("pytest", "-v", "-n", num_processes)
