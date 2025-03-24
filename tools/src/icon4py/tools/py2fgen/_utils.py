# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import pathlib
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import click


def format_fortran_code(source: str) -> str:
    """Format fortran code using fprettify.

    Try to find fprettify in PATH -> found by which
    otherwise look in PYTHONPATH
    """
    fprettify_path = shutil.which("fprettify")

    if fprettify_path is None:
        bin_path = pathlib.Path(sys.executable).parent
        fprettify_path = str(bin_path / "fprettify")
    args = [str(fprettify_path)]
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    return p1.communicate(source.encode("UTF-8"))[0].decode("UTF-8").rstrip()


def write_file(string: str, outdir: pathlib.Path, fname: str) -> None:
    path = outdir / fname
    with open(path, "w") as f:
        f.write(string)


def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with a given name and log level."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def parse_comma_separated_list(_: click.Context, __: click.Parameter, value: str) -> list[str]:
    # Used as `click.argument` callback
    # Splits the input string by commas and strips any leading/trailing whitespace from the strings
    return [item.strip() for item in value.split(",")]
