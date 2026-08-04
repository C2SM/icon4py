# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import abc
import dataclasses
import datetime as dt
import enum
import logging
import pathlib
import statistics
import timeit
import uuid
from collections.abc import Sequence
from typing import Any, Final, TypeAlias

from icon4py.model.common import exceptions, time
from icon4py.model.common.components import monitor
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base, vertical as v_grid
from icon4py.model.common.grid.vertical import VerticalGrid
from icon4py.model.common.io import (
    cf_utils,
    distributed,
    netcdf_writers,
    ugrid,
    writers,
    zarr_writers,
)
from icon4py.model.common.io.writers import GlobalFileAttributes


log = logging.getLogger(__name__)


#: Output schedule given as either a number of model steps or a simulation-time delta.
#: A time delta is normalized to a number of steps internally (using the model time step),
#: so the schedule is always evaluated in steps.
OutputInterval: TypeAlias = time.RelativeTime | time.NumTimeSteps  # noqa: UP040


class OutputBackend(enum.StrEnum):
    """File format an output field group is written in."""

    NETCDF = "netcdf"
    ZARR = "zarr"


class OutputMode(enum.StrEnum):
    """How the ranks of a distributed run write an output field group.

    GATHER: owned entries of all ranks are collected and written by the root rank.
    DISTRIBUTED: every rank writes its owned entries itself, into a rank-contiguous
    block of a shared store. Single-rank runs write the full state either way.
    """

    GATHER = "gather"
    DISTRIBUTED = "distributed"


#: File suffix per output backend (zarr "files" are directories).
FILE_SUFFIXES: Final[dict[OutputBackend, str]] = {
    OutputBackend.NETCDF: ".nc",
    OutputBackend.ZARR: ".zarr",
}

#: Timed phases of a capture step: "distribute" (collective gather / halo stripping)
#: and "write" (file/store output).
PHASE_DISTRIBUTE: Final[str] = "distribute"
PHASE_WRITE: Final[str] = "write"


def _interval_in_steps(output_interval: OutputInterval, dtime: time.RelativeTime) -> int:
    """Normalize an output interval to a number of model steps."""
    if isinstance(output_interval, time.RelativeTime):
        if output_interval < dtime:
            raise exceptions.InvalidConfigError(
                f"Output interval {output_interval} is shorter than the model time step {dtime}."
            )
        if output_interval % dtime:
            raise exceptions.InvalidConfigError(
                f"Output interval {output_interval} is not a multiple of the model "
                f"time step {dtime}."
            )
        return int(output_interval // dtime)
    return output_interval


class Config(abc.ABC):
    """
    Base class for all config classes.

    # TODO(halungge): Need to visit this, when we address configuration
    """

    def __str__(self) -> str:
        return f"instance of {self.__class__}(Config)"

    @abc.abstractmethod
    def validate(self) -> None:
        """
        Validate the config.

        Raises:
            InvalidConfigError: if the config is invalid
        """

        pass


@dataclasses.dataclass(frozen=True)
class FieldGroupIOConfig(Config):
    """
    Structured config for IO of a field group.

    Field group is a number of fields that are output at the same time intervals on the same grid
    (can be any horizontal dimension) and vertical levels.

    """

    filename: str
    variables: list[str]
    #: Output schedule: either a number of model steps (``int``) or a simulation-time
    #: delta (``datetime.timedelta``, must be a multiple of the model time step).
    #: Defaults to every step.
    output_interval: OutputInterval = time.NumTimeSteps(1)
    timesteps_per_file: int = 10
    #: File format of the group's files; the matching value string is also accepted.
    backend: OutputBackend = OutputBackend.ZARR
    #: Write strategy of distributed runs (see ``OutputMode``); the matching value
    #: string is also accepted. Distributed netCDF needs an MPI-parallel netCDF4
    #: installation in multi-rank runs (checked when the writer is created, see
    #: ``netcdf_writers.NETCDFWriter``).
    mode: OutputMode = OutputMode.DISTRIBUTED
    #: Entries per chunk along the horizontal (cell/edge/vertex) axes. Default (None):
    #: one chunk per rank block in distributed mode; otherwise the whole axis (zarr)
    #: or the library default (netCDF).
    horizontal_chunk_size: int | None = None
    #: Zarr only: entries per shard along the horizontal axes, grouping whole chunks
    #: into one storage file each; must be a multiple of ``horizontal_chunk_size``.
    #: Default (None): no sharding, one file per chunk.
    horizontal_shard_size: int | None = None
    nc_title: str = "ICON4Py Simulation"
    nc_comment: str = "ICON inspired code in Python and GT4Py"

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "backend", OutputBackend(self.backend))
        except ValueError as err:
            raise exceptions.InvalidConfigError(
                f"Invalid output 'backend': {self.backend!r}; "
                f"valid values are: {', '.join(b.value for b in OutputBackend)}."
            ) from err
        try:
            object.__setattr__(self, "mode", OutputMode(self.mode))
        except ValueError as err:
            raise exceptions.InvalidConfigError(
                f"Invalid output 'mode': {self.mode!r}; "
                f"valid values are: {', '.join(m.value for m in OutputMode)}."
            ) from err
        self.validate()

    @property
    def block_alignment(self) -> int:
        """Granularity the distributed-mode rank-block size must be a multiple of.

        The shard (or chunk) size of the group's store layout; 1 when neither is
        configured (any block size aligns).
        """
        return self.horizontal_shard_size or self.horizontal_chunk_size or 1

    def _validate_horizontal_chunking(self) -> None:
        for name in ("horizontal_chunk_size", "horizontal_shard_size"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise exceptions.InvalidConfigError(
                    f"Invalid '{name}': must be a positive integer, got {value!r}."
                )
        if self.horizontal_shard_size is None:
            return
        if self.backend != OutputBackend.ZARR:
            raise exceptions.InvalidConfigError(
                "Invalid 'horizontal_shard_size': sharding is only supported by the 'zarr' backend."
            )
        if self.horizontal_chunk_size is None:
            raise exceptions.InvalidConfigError(
                "Invalid 'horizontal_shard_size': requires 'horizontal_chunk_size' "
                "(a shard groups whole chunks)."
            )
        if self.horizontal_shard_size % self.horizontal_chunk_size != 0:
            raise exceptions.InvalidConfigError(
                f"Invalid 'horizontal_shard_size': {self.horizontal_shard_size} is not "
                f"a multiple of 'horizontal_chunk_size' ({self.horizontal_chunk_size})."
            )

    def _validate_filename(self) -> None:
        if not self.filename:
            raise exceptions.InvalidConfigError("Output filename is missing.")
        if self.filename.startswith("/"):
            raise exceptions.InvalidConfigError(
                f"Filename may not be an absolute path: {self.filename}."
            )

    def validate(self) -> None:
        # bool is a subclass of int, but is not a valid interval
        if isinstance(self.output_interval, bool) or not isinstance(
            self.output_interval, OutputInterval
        ):
            raise exceptions.InvalidConfigError(
                f"Output interval must be of type {OutputInterval}: {self.output_interval!r}."
            )
        positive = (
            self.output_interval > time.RelativeTime(0)
            if isinstance(self.output_interval, time.RelativeTime)
            else self.output_interval > 0
        )
        if not positive:
            raise exceptions.InvalidConfigError(
                f"Output interval must be positive: {self.output_interval!r}."
            )
        if not self.variables:
            raise exceptions.InvalidConfigError("No variables provided for output.")
        self._validate_horizontal_chunking()
        self._validate_filename()


@dataclasses.dataclass(frozen=True)
class IOConfig(Config):
    """
    Structured and hierarchical config for IO.

    Holds some general configuration and a collection of configurations for each field group.

    """

    output_path: str = "./output/"
    field_groups: Sequence[FieldGroupIOConfig] = ()
    #: Unit and calendar of the time coordinate in the data files.
    time_units: str = cf_utils.DEFAULT_TIME_UNIT
    calendar: str = cf_utils.DEFAULT_CALENDAR

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.field_groups:
            log.warning("No field configurations provided for output")
        else:
            for field_config in self.field_groups:
                field_config.validate()


class IOMonitor(monitor.Monitor):
    """
    Composite Monitor for all IO groups.

    In a distributed run (multi-rank ``process_props``) the decomposition info is
    required: each field group writes through the output distribution of its configured
    ``mode`` (see ``OutputMode``). ``store`` is then collective on the communicator in
    ``process_props``.
    """

    def __init__(
        self,
        *,
        config: IOConfig,
        vertical_size: v_grid.VerticalGrid,
        horizontal_size: base.HorizontalGridSize,
        grid_file_name: pathlib.Path,
        grid_id: uuid.UUID,
        dtime: time.RelativeTime,
        process_props: decomposition.ProcessProperties | None = None,
        decomposition_info: decomposition.DecompositionInfo | None = None,
    ):
        self.config = config
        # ``grid_file_name`` is the source grid NetCDF, used solely to regenerate the UGRID
        # topology file (`_write_ugrid`); the grid identity comes from ``grid_id`` (the
        # ``Grid`` object), not from the file.
        # TODO(kotsaloscv): build the UGRID topology from ``Grid``/``GridGeometry`` so the
        # monitor no longer needs the source file path at all.
        self._grid_file = grid_file_name
        self._process_props = (
            process_props
            if process_props is not None
            else decomposition.SingleNodeProcessProperties()
        )
        self._decomposition_info = decomposition_info
        self._horizontal_size = horizontal_size
        self._distributions: dict[tuple[OutputMode, int], distributed.OutputDistribution] = {}
        self._timings_logged = False
        self._initialize_output()
        self._group_monitors = [
            FieldGroupMonitor(
                config=conf,
                vertical=vertical_size,
                distribution=self._get_distribution(conf.mode, conf.block_alignment),
                process_props=self._process_props,
                grid_id=grid_id,
                output_path=self._output_path,
                dtime=dtime,
                time_units=config.time_units,
                calendar=config.calendar,
            )
            for conf in config.field_groups
        ]

    def _get_distribution(
        self, mode: OutputMode, block_alignment: int
    ) -> distributed.OutputDistribution:
        """Build (or reuse) the output distribution of a mode; shared between groups.

        Only the rank-block distribution depends on ``block_alignment`` (see
        ``RankBlockDistribution``); for other modes it is normalized to 1 so groups
        share one instance.
        """
        if self._process_props.is_single_rank() or mode == OutputMode.GATHER:
            block_alignment = 1
        key = (mode, block_alignment)
        if key not in self._distributions:
            if self._process_props.is_single_rank():
                distribution: distributed.OutputDistribution = distributed.SingleNodeDistribution(
                    self._horizontal_size
                )
            elif self._decomposition_info is None:
                raise exceptions.InvalidConfigError(
                    "Output in a distributed run requires 'decomposition_info'."
                )
            elif mode == OutputMode.GATHER:
                distribution = distributed.GatherDistribution(
                    self._process_props, self._decomposition_info
                )
            else:
                distribution = distributed.RankBlockDistribution(
                    self._process_props, self._decomposition_info, block_alignment=block_alignment
                )
            self._distributions[key] = distribution
        return self._distributions[key]

    def _initialize_output(self) -> None:
        self._create_output_dir()
        self._write_ugrid()

    def _create_output_dir(self) -> None:
        path = pathlib.Path(self.config.output_path)
        # The directory may already exist: in the driver it is created upfront by
        # ``prepare_output_directory`` (which timestamps it if it already existed). Existing
        # *files* are kept safe though -- ``FieldGroupMonitor._init_dataset`` refuses to
        # overwrite an existing data file, so a rerun into a populated dir fails loudly.
        path.mkdir(parents=True, exist_ok=True)
        self._output_path = path

    def _write_ugrid(self) -> None:
        # the UGRID file is derived from the (global) grid file, identical on all ranks:
        # written once by the root rank; the verdict is broadcast so all ranks raise
        # together instead of hanging in the next collective
        failure: Exception | None = None
        if self._process_props.rank == 0:
            try:
                ugrid.IconUGridWriter(self._grid_file, self._output_path)(validate=True)
            except Exception as err:
                failure = err
        message = str(failure) if failure is not None else None
        if not self._process_props.is_single_rank():
            message = self._process_props.comm.bcast(message, root=0)
        if message is not None:
            raise RuntimeError(
                f"Writing the UGRID topology file for '{self._grid_file}' failed: {message}"
            ) from failure

    @property
    def path(self) -> pathlib.Path:
        return self._output_path

    def store(
        self, state: dict, model_time: dt.datetime, *args: Any, **kwargs: dict[str, Any]
    ) -> None:
        for m in self._group_monitors:
            m.store(state, model_time, *args, **kwargs)

    def close(self) -> None:
        """Close all field-group writers.

        Safe to call from error paths: no communication of its own. Exception: closing
        a parallel netCDF file is collective, but IO-layer errors raise on all ranks
        together (see e.g. ``FieldGroupMonitor._refuse_to_overwrite``), so the ranks
        still close in lockstep.
        """
        for m in self._group_monitors:
            m.close()

    def captures_next_store(self) -> bool:
        """Whether any field group captures at the next ``store`` call.

        A pure predicate (no counter changes), identical on all ranks: callers may
        skip assembling the output state for steps nobody captures -- ``store`` must
        still be called every step (it advances the schedule counters).
        """
        return any(m.captures_next_store() for m in self._group_monitors)

    def report_timings(self) -> None:
        """Log the accumulated output overhead of this rank, per field group and phase.

        Deliberately communication-free: a reduction over the ranks would leave the
        surviving ranks blocked in a collective when one rank failed earlier, and
        timing output is not worth a hang. Every rank logs its own totals; whether
        the non-root lines appear is up to the logging configuration (the driver's
        ``ParallelLogger`` filter passes INFO records on the root rank only, so a
        default driver run reports the root's timings whatever the rank count).
        """
        if self._timings_logged:
            return
        self._timings_logged = True
        for m in self._group_monitors:
            for phase in (PHASE_DISTRIBUTE, PHASE_WRITE):
                seconds = m.phase_seconds[phase]
                if not seconds:
                    continue
                log.info(
                    f"output timings of group '{m.config.filename}', phase '{phase}' on "
                    f"rank {self._process_props.rank}: total {sum(seconds):.6f} s, mean "
                    f"{statistics.mean(seconds):.6f} s over {len(seconds)} capture(s)"
                )


def generate_name(fname: str, counter: int, suffix: str) -> str:
    """File name numbered by the rollover counter, e.g. ``output_0001.zarr``."""
    stem = fname.split(".", maxsplit=1)[0]
    return f"{stem}_{counter:0>4}{suffix}"


class FieldGroupMonitor(monitor.Monitor):
    """
    Monitor for a group of fields.

    This monitor is responsible for storing a group of fields that are output at the same time intervals.

    With a multi-rank ``process_props``, construction is collective on its communicator
    (the file attributes are broadcast so all ranks carry identical values), just like
    ``store`` is.
    """

    def __init__(
        self,
        *,
        config: FieldGroupIOConfig,
        vertical: VerticalGrid,
        distribution: distributed.OutputDistribution,
        grid_id: uuid.UUID,
        dtime: time.RelativeTime,
        process_props: decomposition.ProcessProperties | None = None,
        time_units: str = cf_utils.DEFAULT_TIME_UNIT,
        calendar: str = cf_utils.DEFAULT_CALENDAR,
        output_path: pathlib.Path = pathlib.Path(__file__).parent,
    ):
        self.config = config
        self._time_properties = writers.TimeProperties(time_units, calendar)
        self._vertical_size = vertical
        self._distribution = distribution
        self._process_props = (
            process_props
            if process_props is not None
            else decomposition.SingleNodeProcessProperties()
        )
        # TODO(halungge): 'history' is actually the path to the binary in ICON, not the
        #   output path
        history = output_path.absolute().as_posix() + " " + dt.datetime.now().isoformat()
        if not self._process_props.is_single_rank():
            # rank-local timestamps diverge; in parallel netCDF mode every rank writes
            # the global attributes (collective metadata operations), which requires
            # identical values on all ranks
            history = self._process_props.comm.bcast(history, root=0)
        self._global_attrs: GlobalFileAttributes = {
            "Conventions": "CF-1.7",  # TODO(halungge): check changelog? latest version is 1.11
            "title": config.nc_title,
            "comment": config.nc_comment,
            "institution": "ETH Zurich and MeteoSwiss",
            "source": "https://icon4py.github.io",
            "history": history,
            "references": "https://icon4py.github.io",
            "uuidOfHGrid": grid_id,
        }
        self._field_names = config.variables
        self._handle_output_path(output_path, config.filename)
        self._output_interval_steps = _interval_in_steps(config.output_interval, dtime)
        self._step_counter = 0
        self._file_counter = 0
        self._current_timesteps_in_file = 0
        self._dataset: writers.FieldWriter | None = None
        self._phase_seconds: dict[str, list[float]] = {PHASE_DISTRIBUTE: [], PHASE_WRITE: []}

    @property
    def output_path(self) -> pathlib.Path:
        return self._output_path

    def _handle_output_path(self, output_path: pathlib.Path, filename: str) -> None:
        file = output_path.joinpath(filename).absolute()
        path = file.parent
        path.mkdir(parents=True, exist_ok=True)
        self._output_path = path
        self._file_name_pattern = file.name

    def _next_file_path(self) -> pathlib.Path:
        """Path of the file numbered by the current file counter."""
        filename = generate_name(
            self._file_name_pattern, self._file_counter, FILE_SUFFIXES[self.config.backend]
        )
        return self._output_path.joinpath(filename)

    def _refuse_to_overwrite(self, filename_path: pathlib.Path) -> None:
        """Fail loudly, on every rank, if the next output file already exists.

        The per-run file counter restarts at 0, so file names (``..._0001.nc``) would
        collide with -- and silently overwrite -- output from a previous run sharing this
        directory. Refuse to overwrite so prior results are never lost. The check is
        collective: only the root rank looks at the filesystem and its verdict is
        broadcast, so all ranks raise together -- a root-only raise would leave the
        other ranks blocked in the next collective.
        TODO (jcanton): take care of this when implementing restart
        """
        file_exists = self._process_props.rank == 0 and filename_path.exists()
        if not self._process_props.is_single_rank():
            file_exists = self._process_props.comm.bcast(file_exists, root=0)
        if file_exists:
            raise exceptions.InvalidConfigError(
                f"Output file '{filename_path}' already exists; refusing to overwrite output "
                f"from a previous run. Use a fresh output directory."
            )

    def _init_dataset(self, vertical_params: v_grid.VerticalGrid) -> None:
        """Initialise the dataset with global attributes and dimensions.

        TODO(halungge): as long as we have no terrain it is probably ok to take vct_a as vertical
                          coordinate once there is terrain k-heights become [horizontal, vertical ] field

        """
        if self._dataset is not None:
            self._dataset.close()
        filename_path = self._next_file_path()
        df: writers.FieldWriter
        if self.config.backend == OutputBackend.NETCDF:
            df = netcdf_writers.NETCDFWriter(
                file_name=filename_path,
                vertical=vertical_params,
                horizontal=self._distribution.output_horizontal_size,
                time_properties=self._time_properties,
                global_attrs=self._global_attrs,
                rank_blocks=self._distribution.rank_blocks,
                process_props=self._process_props,
                horizontal_chunk_size=self.config.horizontal_chunk_size,
            )
        else:
            df = zarr_writers.ZarrWriter(
                file_name=filename_path,
                vertical=vertical_params,
                horizontal=self._distribution.output_horizontal_size,
                time_properties=self._time_properties,
                global_attrs=self._global_attrs,
                rank_blocks=self._distribution.rank_blocks,
                process_props=self._process_props,
                horizontal_chunk_size=self.config.horizontal_chunk_size,
                horizontal_shard_size=self.config.horizontal_shard_size,
            )
        df.initialize_dataset()
        self._dataset = df

    def store(
        self, state: dict, model_time: dt.datetime, *args: Any, **kwargs: dict[str, Any]
    ) -> None:
        """Pick fields from the state dictionary to be written to disk.

        In a distributed run this is collective: every rank must call it at every step
        (the distribution communicates at capture steps). File and step counters advance
        identically on all ranks, including ranks that do not write.

        Args:
            state: dict  model state dictionary
            model_time: the current time step of the simulation
        """
        self._step_counter += 1
        if not self._at_capture_time():
            return
        # TODO(halungge): this should do a deep copy of the data once IO becomes
        #   asynchronous (the gather/halo-strip paths already copy, the single-node
        #   path writes synchronously before the state is mutated)
        try:
            state_to_store = {field: state[field] for field in self._field_names}
        except KeyError as e:
            log.error(f"Field '{e.args[0]}' is missing in state.")
            self.close()
            raise exceptions.IncompleteStateError(e.args[0]) from e

        log.info(f"Storing fields {state_to_store.keys()} at {model_time}")

        start = timeit.default_timer()
        prepared_state = self._distribution.prepare(state_to_store)
        self._phase_seconds[PHASE_DISTRIBUTE].append(timeit.default_timer() - start)

        new_file = self._do_initialize_new_file()
        if new_file:
            self._file_counter += 1
            self._refuse_to_overwrite(self._next_file_path())

        if self._distribution.writes_output:
            assert prepared_state is not None
            start = timeit.default_timer()
            if new_file:
                self._init_dataset(self._vertical_size)
            self._append_data(prepared_state, model_time)
            self._phase_seconds[PHASE_WRITE].append(timeit.default_timer() - start)

        self._update_current_file_count()
        if self._is_file_limit_reached():
            self.close()

    def _update_current_file_count(self) -> None:
        self._current_timesteps_in_file = self._current_timesteps_in_file + 1

    def _do_initialize_new_file(self) -> bool:
        return self._current_timesteps_in_file == 0

    def _is_file_limit_reached(self) -> bool:
        return 0 < self.config.timesteps_per_file == self._current_timesteps_in_file

    def _append_data(self, state_to_store: dict, model_time: dt.datetime) -> None:
        assert self._dataset is not None
        self._dataset.append(state_to_store, model_time)

    def _at_capture_time(self) -> bool:
        # fire every N model steps
        return self._step_counter % self._output_interval_steps == 0

    def captures_next_store(self) -> bool:
        """Whether the next ``store`` call falls on a capture step (pure predicate)."""
        return (self._step_counter + 1) % self._output_interval_steps == 0

    @property
    def phase_seconds(self) -> dict[str, list[float]]:
        """Per-phase wall-clock seconds of every capture step (output overhead)."""
        return self._phase_seconds

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
        # reset unconditionally: gather-mode ranks without a writer must keep the same
        # counter values as the writing rank
        self._current_timesteps_in_file = 0
