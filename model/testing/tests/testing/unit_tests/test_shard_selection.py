# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest


pytest_plugins = ["pytester"]


def _collected(pytester, *args):
    result = pytester.runpytest(
        "-p", "icon4py.model.testing.pytest_hooks", "--collect-only", "-q", *args
    )
    return [line for line in result.outlines if "::" in line]


def test_shards_partition_the_tests_without_splitting_modules(pytester):
    pytester.makepyfile(
        test_heavy="import pytest\npytestmark = pytest.mark.shard_weight(100)\n"
        "def test_0(): pass\ndef test_1(): pass\n",
        **{
            f"test_light{i}": "".join(f"def test_{j}(): pass\n" for j in range(i + 1))
            for i in range(5)
        },
    )

    everything = _collected(pytester)
    shards = [_collected(pytester, f"--shard={k}/3") for k in (1, 2, 3)]

    assert len(everything) == 17
    assert sorted(nodeid for shard in shards for nodeid in shard) == sorted(everything)
    modules = [{nodeid.split("::")[0] for nodeid in shard} for shard in shards]
    assert sum(len(shard_modules) for shard_modules in modules) == len(set().union(*modules))
    assert {"test_heavy.py"} in modules


@pytest.mark.parametrize("spec", ["0/2", "3/2", "1-2"])
def test_shard_rejects_invalid_spec(pytester, spec):
    pytester.makepyfile("def test_0(): pass\n")

    result = pytester.runpytest("-p", "icon4py.model.testing.pytest_hooks", f"--shard={spec}")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
