# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import pathlib
import tarfile

from icon4py.model.testing import config, locking
from icon4py.model.testing.definitions import Experiment


def experiment_name_with_version(experiment: Experiment) -> str:
    """Generate experiment name with version suffix.

    Args:
        experiment: Experiment object

    Returns:
        Name with version in format '{name}_v{version:02d}'
    """
    return f"{experiment.name}_v{experiment.version:02d}"


def experiment_archive_filename(experiment: Experiment) -> str:
    """Generate archive filename for an experiment.

    Args:
        experiment: Experiment object

    Returns:
        Archive filename in format '{name}_v{version:02d}.tar.gz'
    """
    return f"{experiment_name_with_version(experiment)}.tar.gz"


def download_and_extract(uri: str, dst: pathlib.Path, data_file: str = "downloaded.tar.gz") -> None:
    """
    Download data archive from remote server.

    Downloads a tar file at `uri` and extracts it.
    Args:
        uri: download url for archived data
        destination: the archive is extracted at this path
        data_file: filename of the downloaded archive, the archive is removed after download
    """
    dst.mkdir(parents=True, exist_ok=True)
    try:
        import wget  # type: ignore[import-untyped]
    except ImportError as err:
        raise RuntimeError(f"To download data file from {uri}, please install `wget`") from err

    wget.download(uri, out=data_file)
    if not tarfile.is_tarfile(data_file):
        raise OSError(f"{data_file} needs to be a valid tar file")
    with tarfile.open(data_file, mode="r:*") as tf:
        tf.extractall(path=dst)
    pathlib.Path(data_file).unlink(missing_ok=True)


def download_test_data(dst: pathlib.Path, uri: str) -> None:
    if config.ENABLE_TESTDATA_DOWNLOAD:
        # We create and lock the *parent* directory as we later check for existence of `dst`.
        dst.parent.mkdir(parents=True, exist_ok=True)
        with locking.lock(dst.parent):
            if not dst.exists():
                download_and_extract(uri, dst)
    else:
        # If test data download is disabled, we check if the directory exists
        # without locking. We assume the location is managed by the user
        # and avoid locking shared directories (e.g. on CI).
        if not dst.exists():
            raise RuntimeError(f"Test data {dst} does not exist, and downloading is disabled.")
