# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest


pytest_plugins = ["pytester"]

_SUITES = """
import pytest


def _static_variant(request):
    return request.param[1]


class TestAllVariants:
    STATIC_PARAMS = {"none": (), "compile_time_domain": ("a",), "compile_time_vertical": ("b",)}
    static_variant = pytest.fixture(params=STATIC_PARAMS.items(), ids=lambda p: p[0])(_static_variant)

    @pytest.mark.parametrize("flag", [True, False])
    def test_program(self, static_variant, flag):
        pass


class TestWithoutDomainVariant:
    STATIC_PARAMS = {"none": (), "compile_time_vertical": ("b",)}
    static_variant = pytest.fixture(params=STATIC_PARAMS.items(), ids=lambda p: p[0])(_static_variant)

    def test_program(self, static_variant):
        pass


class TestWithoutStaticParams:
    STATIC_PARAMS = None

    def test_program(self):
        pass
"""


def _collected(pytester, *args):
    result = pytester.runpytest(
        "-p", "icon4py.model.testing.pytest_hooks", "--collect-only", "-q", *args
    )
    return sorted(line.split("::", 1)[1] for line in result.outlines if "::" in line)


def test_static_variant_keeps_only_that_variant_of_suites_defining_it(pytester):
    pytester.makepyfile(test_suites=_SUITES)

    assert len(_collected(pytester)) == 9
    assert _collected(pytester, "--static-variant=compile_time_domain") == [
        "TestAllVariants::test_program[compile_time_domain-False]",
        "TestAllVariants::test_program[compile_time_domain-True]",
        "TestWithoutDomainVariant::test_program[compile_time_vertical]",
        "TestWithoutDomainVariant::test_program[none]",
        "TestWithoutStaticParams::test_program",
    ]
