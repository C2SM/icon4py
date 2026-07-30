# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Check downloaded test data against the records that were blessed with it.

The grid, metric and interpolation fields of a published archive must not change under
a version that has already been released. When they do, it means either that the
archive behind a published name was replaced, or that a version was bumped without
re-blessing the data. Either way it is far cheaper to see it here than as an
unexplained failure in an unrelated stencil test.
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import pytest

from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    serialized_data,
)
from icon4py.model.testing.fixtures import (
    download_ser_data,
    experiment,
    experiment_description,
    process_props,
)


if TYPE_CHECKING:
    from icon4py.model.common.decomposition import definitions as decomposition


LOCK_DIR = pathlib.Path(__file__).resolve().parents[3] / "testdata"

_EXPERIMENTS = [
    test_defs.Experiments.EXCLAIM_APE,
    test_defs.Experiments.EXCLAIM_APE_AES,
    test_defs.Experiments.MCH_CH_R04B09,
    test_defs.Experiments.JW,
    test_defs.Experiments.GAUSS3D,
    test_defs.Experiments.WEISMAN_KLEMP_TORUS,
]


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", _EXPERIMENTS, ids=lambda e: e.name)
def test_testdata_matches_lock(
    experiment: test_defs.Experiment,
    experiment_description: test_defs.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
) -> None:
    lock_path = LOCK_DIR / f"{experiment_description.name}.lock.json"
    if not lock_path.is_file():
        pytest.skip(f"No lockfile for '{experiment_description.name}'; run 'serdata bless'.")
    lock = json.loads(lock_path.read_text())

    assert lock["version"] == experiment_description.version, (
        f"'{lock_path.name}' pins v{lock['version']:02d} but the experiment is at "
        f"v{experiment_description.version:02d}. Re-bless the archive after a version bump."
    )

    # The communicator size is a property of the running job, not of the test, so a
    # lockfile that has no entry for it simply does not constrain this run.
    entry = lock["archives"].get(str(process_props.comm_size))
    if entry is None:
        pytest.skip(f"Lockfile has no entry for {process_props.comm_size} rank(s).")

    archive_dir = dt_utils.get_path_for_experiment(experiment_description, process_props)
    problems = serialized_data.verify_against_lock(
        entry, serialized_data.fingerprint_archive(archive_dir)
    )

    assert not problems, "\n".join(
        [f"{len(problems)} pinned record(s) differ from '{lock_path.name}':", *problems[:20]]
    )
