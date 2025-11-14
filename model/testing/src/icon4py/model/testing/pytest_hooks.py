# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import contextlib
import os
import re

import numpy as np
import pytest

from icon4py.model.common import model_backends
from icon4py.model.testing import filters

from gt4py.next import config as gtx_config

__all__ = [
    "pytest_addoption",
    "pytest_benchmark_update_json",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_runtest_setup",
]

_TEST_LEVELS = ("any", "unit", "integration")


def pytest_configure(config):
    config.addinivalue_line("markers", "datatest: this test uses binary data")
    config.addinivalue_line(
        "markers", "with_netcdf: test uses netcdf which is an optional dependency"
    )
    config.addinivalue_line(
        "markers",
        "level(name): marks test as unit or integration tests, mostly applicable where both are available",
    )

    # Check if the --enable-mixed-precision option is set and set the environment variable accordingly
    if config.getoption("--enable-mixed-precision"):
        os.environ["FLOAT_PRECISION"] = "mixed"

    # Handle datatest options: --datatest-only  and --datatest-skip
    if m_option := config.getoption("-m", []):
        m_option = [f"({m_option})"]  # add parenthesis around original k_option just in case
    if config.getoption("--datatest-only"):
        config.option.markexpr = " and ".join(["datatest", *m_option])

    if config.getoption("--datatest-skip"):
        config.option.markexpr = " and ".join(["not datatest", *m_option])


def pytest_addoption(parser: pytest.Parser):
    """Add custom commandline options for pytest."""
    try:
        datatest = parser.getgroup("datatest", "Options for data testing")
        datatest.addoption(
            "--datatest-skip",
            action="store_true",
            default=False,
            help="Skip all data tests",
        )
        datatest.addoption(
            "--datatest-only",
            action="store_true",
            default=False,
            help="Run only data tests",
        )
    except ValueError:
        pass
    with contextlib.suppress(ValueError):
        parser.addoption(
            "--backend",
            action="store",
            default=model_backends.DEFAULT_BACKEND,
            help="GT4Py backend to use when executing stencils. Defaults to roundtrip backend, other options include gtfn_cpu, gtfn_gpu, and embedded",
        )

    with contextlib.suppress(ValueError):
        parser.addoption(
            "--grid",
            action="store",
            help="Grid to use.",
        )

    with contextlib.suppress(ValueError):
        parser.addoption(
            "--enable-mixed-precision",
            action="store_true",
            help="Switch unit tests from double to mixed-precision",
            default=False,
        )

    with contextlib.suppress(ValueError):
        parser.addoption(
            "--level",
            action="store",
            choices=_TEST_LEVELS,
            help="Set level (unit, integration) of the tests to run. Defaults to 'any'.",
            default="any",
        )


def pytest_collection_modifyitems(config, items):
    """Modify collected test items based on command line options."""
    test_level = config.getoption("--level")
    if test_level == "any":
        return
    for item in items:
        if (marker := item.get_closest_marker("level")) is not None:
            assert all(
                level in _TEST_LEVELS for level in marker.args
            ), f"Invalid test level argument on function '{item.name}' - possible values are {_TEST_LEVELS}"
            if test_level not in marker.args:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Selected level '{test_level}' does not match the configured '{marker.args}' level for this test."
                    )
                )


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Apply test item filters as the final test setup step."""

    item_marker_filters = filters.item_marker_filters
    for marker_name in set(m.name for m in item.iter_markers()) & item_marker_filters.keys():
        item_filter = item_marker_filters[marker_name]
        if item_filter.condition(item):
            item_filter.action()


_name_from_fullname_pattern = re.compile(
    r"""
        ::(?P<class>[A-Za-z_]\w*)       # capture class name
        (?::: [A-Za-z_]\w*              # skip method name
        (?:\[(?P<params>[^\]]+)\])? )   # optional parameterization
        """,
    re.VERBOSE,
)


def _name_from_fullname(fullname: str) -> str:
    match = _name_from_fullname_pattern.search(fullname)
    if match is None:
        return fullname  # assume already fixed
    class_name = match.group("class")
    params = match.group("params")
    return f"{class_name}[{params}]" if params else class_name


# pytest benchmark hook, see:
#     https://pytest-benchmark.readthedocs.io/en/latest/hooks.html#pytest_benchmark.hookspec.pytest_benchmark_update_json
def pytest_benchmark_update_json(output_json):
    """
    Replace 'fullname' of pytest benchmarks with a shorter name for better readability in bencher.

    Note:
    Currently works only for 'StencilTest's as they have the following fixed structure:
      '<path>::<class_name>::test_stencil[<variant>]'.
    """

    for bench in output_json["benchmarks"]:
        bench["fullname"] = _name_from_fullname(bench["fullname"])
        # if GT4Py metrics collection is enabled, replace the benchmark stats used by `bencher` with the GT4Py metrics stats
        # to avoid reporting python overheads in `bencher` so that the results are comparable to the Fortran stencil benchmarks
        if gtx_config.COLLECT_METRICS_LEVEL > 0:
            gt4py_metrics_runtimes = bench.get("extra_info", {}).get("gtx_metrics", [])
            assert len(gt4py_metrics_runtimes) > 0, "No GT4Py metrics collected despite COLLECT_METRICS_LEVEL > 0"
            bench["stats"]["mean"] = np.mean(gt4py_metrics_runtimes)
            bench["stats"]["median"] = np.median(gt4py_metrics_runtimes)
            bench["stats"]["stddev"] = np.std(gt4py_metrics_runtimes)
            bench["stats"]["q1"] = np.percentile(gt4py_metrics_runtimes, 25)
            bench["stats"]["q3"] = np.percentile(gt4py_metrics_runtimes, 75)
            bench["stats"]["iqr"] = bench["stats"]["q3"] - bench["stats"]["q1"]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Gather GT4Py timer metrics from benchmark fixture and add them to the test report.
    """
    outcome = yield
    report = outcome.get_result()
    if call.when == "call":
        benchmark = item.funcargs.get("benchmark", None)
        if benchmark and hasattr(benchmark, "extra_info"):
            info = benchmark.extra_info.get("gtx_metrics", None)
            if info:
                filtered_benchmark_name = benchmark.name.split("test_Test")[-1]
                # Combine the benchmark name in a readable form with the gtx_metrics data
                report.sections.append(("benchmark-extra", tuple([filtered_benchmark_name, info])))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Add a custom section to the terminal summary with GT4Py timer metrics from benchmarks.
    """
    # Gather gtx_metrics
    benchmark_gtx_metrics = []
    for outcome in ("passed", "failed", "skipped"):
        all_reports = terminalreporter.stats.get(outcome, [])
        for report in all_reports:
            for secname, info in getattr(report, "sections", []):
                if secname == "benchmark-extra":
                    benchmark_gtx_metrics.append(info)
    # Calculate the maximum length of benchmark names for formatting
    max_name_len = 0
    for benchmark_name, _ in benchmark_gtx_metrics:
        max_name_len = max(len(benchmark_name), max_name_len)
    # Print the GT4Py timer report table
    if benchmark_gtx_metrics:
        terminalreporter.ensure_newline()
        header = f"{'Benchmark Name':<{max_name_len}} | {'Mean (s)':>10} | {'Median (s)':>10} | {'Std Dev':>10} | {'Runs':>4}"
        title = " GT4Py Timer Report "
        sep_len = max(0, len(header) - len(title))
        left = sep_len // 2
        right = sep_len - left
        terminalreporter.line("-" * left + title + "-" * right, bold=True, blue=True)
        terminalreporter.line(header)
        terminalreporter.line("-" * len(header), blue=True)
        import numpy as np

        for benchmark_name, gtx_metrics in benchmark_gtx_metrics:
            terminalreporter.line(
                f"{benchmark_name:<{max_name_len}} | {np.mean(gtx_metrics):>10.8f} | {np.median(gtx_metrics):>10.8f} | {np.std(gtx_metrics):>10.8f} | {len(gtx_metrics):>4}"
            )
        terminalreporter.line("-" * len(header), blue=True)
