# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt
import functools
import importlib.util
import logging
import pathlib
import types
from typing import Any, Final, Self

import netCDF4 as nc
import numpy as np
import xarray as xr

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base, vertical as v_grid
from icon4py.model.common.io import cf_utils, distributed, writers
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)

#: The steps enabling MPI-parallel netCDF output, spelled out wherever an installation
#: without it is rejected (see ``missing_parallel_support``).
PARALLEL_INSTALL_HINT: Final[str] = (
    "To enable parallel netCDF: (1) provide MPI-enabled netCDF-C and HDF5 libraries "
    "(HPC environment modules, spack, or the conda-forge 'netcdf4=*=mpi_*' builds, "
    "which already include steps 2-3); (2) install 'mpi4py' and the build requirements "
    "('Cython', 'numpy', 'setuptools', 'setuptools_scm'), then build the Python "
    "package against the parallel libraries: 'HDF5_DIR=<hdf5-prefix> "
    "NETCDF4_DIR=<netcdf-c-prefix> pip install --no-binary netcdf4 "
    "--no-build-isolation --force-reinstall netcdf4' (mpi4py must be importable "
    'during the build, hence --no-build-isolation); (3) verify that "python -c '
    "'import netCDF4; print(netCDF4.__has_parallel4_support__)'\" prints 1."
)

#: netCDF-C rejects HDF5 chunks of 4 GiB or more ("NetCDF: Bad chunk sizes").
_MAX_CHUNK_BYTES: Final[int] = 2**32 - 1


def _bounded_middle_chunks(
    middle_shape: tuple[int, ...], horizontal_chunk: int, itemsize: int
) -> tuple[int, ...]:
    """Chunk sizes of the dimensions between time and the horizontal axis.

    The horizontal axis must keep exactly one chunk per rank block (concurrent writes
    of different ranks stay in disjoint chunks), so only the middle (e.g. vertical)
    chunk sizes can shrink to respect the HDF5 chunk size limit -- the equivalent zarr
    layout has no such limit.

    Raises:
        RuntimeError: if the horizontal chunk alone already exceeds the limit.
    """
    budget = _MAX_CHUNK_BYTES // (itemsize * max(horizontal_chunk, 1))
    if budget < 1:
        raise RuntimeError(
            f"A rank block of {horizontal_chunk} entries of {itemsize} bytes exceeds "
            f"the HDF5 chunk size limit ({_MAX_CHUNK_BYTES} bytes). Write with more "
            "ranks (smaller per-rank blocks), or use the 'zarr' backend or the "
            "'gather' mode."
        )
    chunks: list[int] = []
    for size in reversed(middle_shape):
        take = min(size, budget)
        chunks.append(take)
        budget //= take
        budget = max(budget, 1)
    chunks.reverse()
    return tuple(chunks)


def build_description() -> str:
    """One-line description of the installed netCDF4 build, for logs and error messages."""
    return (
        f"netCDF4 {nc.__version__}, netcdf-c {nc.__netcdf4libversion__}, "
        f"HDF5 {nc.__hdf5libversion__}, parallel support: "
        f"{bool(getattr(nc, '__has_parallel4_support__', False))}"
    )


def missing_parallel_support() -> str | None:
    """Why the installed netCDF4 package cannot write in parallel; None if it can.

    MPI-parallel writes of the NETCDF4 file format need the ``netCDF4`` package
    compiled against MPI-enabled netCDF-C and HDF5 libraries (reported by
    ``netCDF4.__has_parallel4_support__``) plus ``mpi4py``. The wheels on PyPI are
    serial builds -- parallel HDF5 cannot be shipped in a portable wheel -- so a plain
    ``pip install netcdf4`` never has parallel support, whatever its version.
    """
    if not getattr(nc, "__has_parallel4_support__", False):
        return (
            f"the installed netCDF4 package ({build_description()}) is a serial build "
            "(netCDF4.__has_parallel4_support__ is false), as all PyPI wheels are"
        )
    if importlib.util.find_spec("mpi4py") is None:
        return "the 'mpi4py' package is not installed"
    return None


class NETCDFWriter:
    """
    Writer for netcdf files.

    Writes a netcdf file using netcdf4-python directly. Currently, this seems to be the only way that we can
      - append time slices to a variable already present in the file. (Xarray.to_netcdf does not support this https://github.com/pydata/xarray/issues/1672)

    Serial mode (``rank_blocks is None``): whole time slices are written by a single
    process -- a single-rank run, or the root rank of gathered output
    (``distributed.GatherDistribution``).

    Rank-block mode (``rank_blocks`` given): every rank writes only its rank-contiguous
    block of the horizontal axes (see ``distributed.RankBlock``), mirroring the layout
    of the rank-block zarr store: padded horizontal axes chunked with exactly one chunk
    per rank block, ``global_index_<dim>`` coordinates mapping file positions to the
    undecomposed global grid (-1 marks padding) and data padding reading as NaN for
    floating dtypes. On a multi-rank communicator this requires an MPI-parallel netCDF4
    installation (see ``missing_parallel_support``; checked at construction and at file
    open): the ranks share one file opened with ``parallel=True``, every rank performs
    the metadata operations (collective in netCDF), and variables touching the
    unlimited time dimension are written in collective mode -- each rank covering its
    full block, padding included, so every rank participates in every collective write
    even when it owns no entries.
    """

    def __init__(
        self,
        *,
        file_name: pathlib.Path | str,
        vertical: v_grid.VerticalGrid,
        horizontal: base.HorizontalGridSize,
        time_properties: writers.TimeProperties,
        global_attrs: writers.GlobalFileAttributes,
        rank_blocks: dict[str, distributed.RankBlock] | None = None,
        process_props: decomposition.ProcessProperties | None = None,
    ):
        self._file_name = str(file_name)
        self._time_properties = time_properties
        self._vertical_params = vertical
        self._horizontal_size = horizontal
        self.attrs = global_attrs
        self._rank_blocks = rank_blocks
        self._process_props = (
            process_props
            if process_props is not None
            else decomposition.SingleNodeProcessProperties()
        )
        self.dataset: nc.Dataset | None = None
        if self._is_parallel():
            reason = missing_parallel_support()
            if reason is not None:
                raise RuntimeError(
                    f"Cannot write '{self._file_name}' in parallel "
                    f"({self._process_props.comm_size} ranks share one netCDF file): "
                    f"{reason}. {PARALLEL_INSTALL_HINT}"
                )

    def __getitem__(self, item: str) -> str:
        assert self.dataset is not None
        return self.dataset.getncattr(item)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close()

    @functools.cached_property
    def num_levels(self) -> int:
        return self._vertical_params.interface_physical_height.ndarray.shape[0] - 1

    @functools.cached_property
    def num_interfaces(self) -> int:
        return self._vertical_params.interface_physical_height.ndarray.shape[0]

    def _is_parallel(self) -> bool:
        """Whether the ranks of a multi-rank communicator share this file.

        Rank-block layout alone does not imply parallel access: on a single-rank
        communicator the same layout is written through a plain serial file handle
        (which is also what makes the layout testable without a parallel build).
        """
        return self._rank_blocks is not None and not self._process_props.is_single_rank()

    def _is_root(self) -> bool:
        return self._process_props.rank == 0

    def _horizontal_write_range(self, dim_name: str) -> slice:
        """This rank's full horizontal block (the whole file axis in serial mode).

        Data writes cover the whole block including its padding (see ``_pad_to_block``),
        never just the ``count`` owned entries: netCDF4-python silently skips zero-size
        writes, so a rank owning nothing would otherwise stay out of a collective write
        the other ranks are waiting in.
        """
        if self._rank_blocks is None:
            return slice(None)
        block = self._rank_blocks[dim_name]
        return slice(block.start, block.start + block.chunk)

    def _pad_to_block(self, dim_name: str, data: np.ndarray) -> np.ndarray:
        """Extend owned entries along the last axis to the full rank block.

        The padding value matches what an unwritten entry reads as (NaN for floating
        dtypes -- also the variables' fill value -- and the netCDF default fill value
        otherwise), so written and unwritten padding are indistinguishable.
        """
        assert self._rank_blocks is not None
        block = self._rank_blocks[dim_name]
        pad_value = (
            np.nan
            if np.issubdtype(data.dtype, np.floating)
            else nc.default_fillvals[f"{data.dtype.kind}{data.dtype.itemsize}"]
        )
        padded = np.full((*data.shape[:-1], block.chunk), pad_value, dtype=data.dtype)
        padded[..., : block.count] = data
        return padded

    def _open_dataset(self) -> nc.Dataset:
        if not self._is_parallel():
            return nc.Dataset(self._file_name, "w", format="NETCDF4", persist=True)
        log.info(f"Opening {self._file_name} for parallel writing ({build_description()})")
        try:
            return nc.Dataset(
                self._file_name,
                "w",
                format="NETCDF4",
                parallel=True,
                comm=self._process_props.comm,
            )
        except (ValueError, OSError) as err:
            # last-resort check: reached when the support flags lie about the build or
            # the file system cannot do MPI-IO
            raise RuntimeError(
                f"Opening '{self._file_name}' for parallel writing failed "
                f"({build_description()}). {PARALLEL_INSTALL_HINT}"
            ) from err

    def initialize_dataset(self) -> None:
        self.dataset = self._open_dataset()
        assert self.dataset is not None
        log.info(f"Creating file {self._file_name} at {self.dataset.filepath()}")
        # metadata operations (attributes, dimensions, variable creation) are collective
        # in parallel mode: every rank performs them, identically; data writes below are
        # restricted to one rank (independent access) or one block per rank instead
        self.dataset.setncatts({k: str(v) for (k, v) in self.attrs.items()})
        ## create dimensions all except time are fixed
        self.dataset.createDimension(writers.TIME, None)
        self.dataset.createDimension(writers.MODEL_LEVEL, self.num_levels)
        self.dataset.createDimension(writers.MODEL_HALF_LEVEL, self.num_interfaces)
        self.dataset.createDimension(writers.CELL, self._horizontal_size.num_cells)
        self.dataset.createDimension(writers.VERTEX, self._horizontal_size.num_vertices)
        self.dataset.createDimension(writers.EDGE, self._horizontal_size.num_edges)
        log.debug(f"Creating dimensions {self.dataset.dimensions} in {self._file_name}")
        # create time variables
        times = self.dataset.createVariable(writers.TIME, "f8", (writers.TIME,))
        times.setncatts(writers.time_attributes(self._time_properties))
        if self._is_parallel():
            # writes touching the unlimited time dimension must be collective
            times.set_collective(True)
        # create vertical coordinates:
        levels = self.dataset.createVariable(writers.MODEL_LEVEL, np.int32, (writers.MODEL_LEVEL,))
        levels.setncatts(writers.LEVEL_ATTRIBUTES)
        half_levels = self.dataset.createVariable(
            writers.MODEL_HALF_LEVEL, np.int32, (writers.MODEL_HALF_LEVEL,)
        )
        half_levels.setncatts(writers.HALF_LEVEL_ATTRIBUTES)
        heights = self.dataset.createVariable("height", np.float64, (writers.MODEL_HALF_LEVEL,))
        heights.setncatts(writers.HEIGHT_ATTRIBUTES)
        if self._is_root():
            # fixed-size coordinates, identical on all ranks: one writer suffices
            levels[:] = np.arange(self.num_levels, dtype=np.int32)
            half_levels[:] = np.arange(self.num_levels + 1, dtype=np.int32)
            heights[:] = data_alloc.as_numpy(self._vertical_params.interface_physical_height)
        self._write_global_index()

    def _write_global_index(self) -> None:
        """Create and fill the ``global_index_<dim>`` coordinates of a rank-block file."""
        if self._rank_blocks is None:
            return
        assert self.dataset is not None
        for dim_name, block in self._rank_blocks.items():
            variable = self.dataset.createVariable(
                f"{writers.GLOBAL_INDEX_PREFIX}_{dim_name}",
                np.int64,
                (dim_name,),
                chunksizes=(block.chunk,),
            )
            variable.setncatts(
                {
                    "units": "1",
                    "long_name": (
                        f"position of each {dim_name} entry in the undecomposed global "
                        f"grid of {block.global_size} entries (-1 marks padding)"
                    ),
                }
            )
            # each rank writes its whole block: the owned global indices followed by
            # explicit -1 padding. The zarr store encodes the -1 as the array's fill
            # value instead; a netCDF _FillValue attribute is decoded as a missing
            # value by xarray, which would turn the integer coordinate into floats on
            # read.
            block_values = np.full((block.chunk,), -1, dtype=np.int64)
            block_values[: block.count] = block.global_index
            variable[block.start : block.start + block.chunk] = block_values

    def append(self, state_to_append: dict[str, xr.DataArray], model_time: dt.datetime) -> None:
        """
        Append the fields to the dataset.

        Appends a time slice of the fields in the state_to_append dictionary to the dataset
        for the `model_time`, expanding the time coordinate by the `model_time`. In
        rank-block mode only this rank's horizontal block is written.

        Args:
            state_to_append: fields to append
            model_time: time of the model state
        """
        assert self.dataset is not None
        canonical_slices: dict[str, xr.DataArray] = {}
        host_data: dict[str, np.ndarray] = {}
        # canonicalize and transfer to host up front: a failure here (unsupported
        # dimensions, device buffer that cannot be converted) must precede any file
        # mutation, or the file would be left with a phantom time slice
        for var_name, new_slice in state_to_append.items():
            canonical_slice = cf_utils.to_canonical_dim_order(new_slice)
            if canonical_slice is None:
                raise ValueError(
                    f"Cannot write field '{var_name}': only fields with a horizontal and a "
                    f"vertical dimension are supported."
                )
            canonical_slices[var_name] = canonical_slice
            host_data[var_name] = data_alloc.as_numpy(canonical_slice.data)
        time = self.dataset[writers.TIME]
        time_pos = len(time)
        # every rank participates (the time variable is collective in parallel mode)
        # and writes the same value
        time[time_pos] = cf_utils.date2num(model_time, units=time.units, calendar=time.calendar)
        for var_name, canonical_slice in canonical_slices.items():
            standard_name = canonical_slice.standard_name
            assert standard_name is not None, f"No standard_name provided for {var_name}."
            existing = writers.filter_by_standard_name(self.dataset.variables, standard_name)
            if not existing:
                variable = self._create_variable(var_name, canonical_slice)
            else:
                variable = next(iter(existing.values()))
                assert len(canonical_slice.dims) == len(variable.dimensions) - 1, (
                    f"Data variable dimensions do not match for {standard_name}."
                )
            dim_name = str(canonical_slice.dims[-1])
            data = host_data[var_name]
            if self._rank_blocks is not None:
                data = self._pad_to_block(dim_name, data)
            horizontal_range = self._horizontal_write_range(dim_name)
            middle = (slice(None),) * (len(canonical_slice.dims) - 1)
            index: tuple[int | slice, ...] = (time_pos, *middle, horizontal_range)
            # in parallel mode this write is collective: every rank must genuinely
            # reach the underlying library call, which is why it covers the full,
            # never empty block -- a zero-size write would be skipped inside
            # netCDF4-python, leaving this rank out of a collective the other ranks
            # are waiting in
            variable[index] = data

    def _create_variable(self, var_name: str, canonical_slice: xr.DataArray) -> Any:
        """Create the array of a data variable (on every rank -- collective in parallel mode)."""
        assert self.dataset is not None
        create_kwargs: dict[str, Any] = {}
        if self._rank_blocks is not None:
            horizontal_name = str(canonical_slice.dims[-1])
            block = self._rank_blocks[horizontal_name]
            # one chunk per rank block along the horizontal axis (the layout of the
            # rank-block zarr store), keeping concurrent writes of different ranks in
            # disjoint chunks; the middle chunk sizes shrink only if the HDF5 chunk
            # size limit demands it. The NaN fill makes any never-written entry of a
            # floating variable read as missing (other dtypes read as the netCDF
            # default fill values).
            middle_chunks = _bounded_middle_chunks(
                canonical_slice.shape[:-1], block.chunk, canonical_slice.dtype.itemsize
            )
            create_kwargs["chunksizes"] = (1, *middle_chunks, max(block.chunk, 1))
            if np.issubdtype(canonical_slice.dtype, np.floating):
                create_kwargs["fill_value"] = np.nan
        dimensions = (writers.TIME, *(str(d) for d in canonical_slice.dims))
        variable = self.dataset.createVariable(
            var_name, canonical_slice.dtype, dimensions, **create_kwargs
        )
        variable.setncatts(writers.data_variable_attributes(canonical_slice))
        if self._is_parallel():
            # writes touching the unlimited time dimension must be collective
            variable.set_collective(True)
        return variable

    def close(self) -> None:
        """Close the file. Collective in parallel mode (an MPI-opened HDF5 file):
        every rank of the communicator must call it."""
        assert self.dataset is not None
        if self.dataset.isopen():
            self.dataset.close()

    @property
    def dims(self) -> dict:
        assert self.dataset is not None
        return self.dataset.dimensions

    @property
    def variables(self) -> dict:
        assert self.dataset is not None
        return self.dataset.variables
