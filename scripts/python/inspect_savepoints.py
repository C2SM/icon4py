#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect serialized ICON savepoint archives.

Summarizes the fields a savepoint holds, so that reference data can be checked for
carrying information at all -- an all-zero array validates nothing -- before a datatest
relies on it. Select the archive by experiment (``-e``, resolved through
``icon4py.model.testing.definitions``) or by a path to a ``ser_data`` directory (``-p``).

    ./scripts/run inspect-savepoints savepoints -e exclaim_ape_aesPhys
    ./scripts/run inspect-savepoints stats -e exclaim_nh_weisman_klemp -v 7 -f tracers
    ./scripts/run inspect-savepoints diff -e exclaim_ape_aesPhys -n aes-graupel-exit -f tracers
    ./scripts/run inspect-savepoints shell -e exclaim_ape_aesPhys
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Final

import numpy as np
import typer


if TYPE_CHECKING:
    from icon4py.model.testing import definitions as test_defs


cli = typer.Typer(name="inspect-savepoints", no_args_is_help=True, help=__doc__)

DEFAULT_PREFIX: Final = "icon_pydycore"

# Fields whose trailing axis indexes the tracer species rather than a grid dimension.
# ICON names all of them after the list they slice: 'tracers', 'tracers_now',
# 'tend_tracers', 'grf_tend_tracers', 'hfl_tracers', ...
TRACER_FIELD_MARKER: Final = "tracers"

# Fields listed per savepoint in the 'savepoints' overview before eliding the rest.
MAX_LISTED_FIELDS: Final = 8


# -- archive selection --


def resolve_experiment(name: str) -> test_defs.ExperimentDescription:
    """Resolve an experiment by its ``Experiments`` attribute name or its ICON name."""
    from icon4py.model.testing import definitions  # noqa: PLC0415 [import-outside-top-level]

    known = [
        (attribute, value)
        for attribute, value in vars(definitions.Experiments).items()
        if isinstance(value, definitions.ExperimentDescription)
    ]
    for attribute, description in known:
        if name.lower() in (attribute.lower(), description.name.lower()):
            return description
    raise typer.BadParameter(
        f"Unknown experiment '{name}'. Known experiments: "
        f"{', '.join(sorted(description.name for _, description in known))}."
    )


def experiment_data_path(
    description: test_defs.ExperimentDescription, comm_size: int
) -> pathlib.Path:
    """Path of the ``ser_data`` directory of an experiment archive."""
    from icon4py.model.testing import (  # noqa: PLC0415 [import-outside-top-level]
        datatest_utils,
        definitions,
    )

    directory = datatest_utils.get_ranked_experiment_name_with_version(description, comm_size)
    return definitions.serialized_data_path() / directory / definitions.SERIALIZED_DATA_SUBDIR


# -- data model --


@dataclasses.dataclass(frozen=True)
class SavepointRef:
    """A savepoint in an archive, identified by its position in the archive order."""

    index: int
    name: str
    date: str | None

    @property
    def label(self) -> str:
        return f"{self.name} @ {self.date}" if self.date else f"{self.name} #{self.index}"


@dataclasses.dataclass(frozen=True)
class FieldStats:
    """Summary of one field, or of one species of a multi-component field."""

    savepoint: str
    date: str | None
    field: str
    component: str | None
    shape: tuple[int, ...]
    min: float
    max: float
    mean: float
    nonzero_fraction: float
    n_nonfinite: int

    @property
    def all_zero(self) -> bool:
        return self.min == 0.0 and self.max == 0.0

    @property
    def full_name(self) -> str:
        return f"{self.field}.{self.component}" if self.component else self.field


def component_labels(field: str, size: int) -> tuple[str, ...] | None:
    """Labels for the trailing axis of *field*, or None if it is not a component axis."""
    if TRACER_FIELD_MARKER not in field:
        return None
    # the serialization order of the tracers, as the model itself defines it
    from icon4py.model.common.states.tracer_states import (  # noqa: PLC0415 [import-outside-top-level]
        _TRACER_FIELDS as known,
    )

    return tuple(known[i] if i < len(known) else f"idx{i}" for i in range(size))


def _transition(before: str | None, after: str | None) -> str | None:
    """Label a pair of endpoints, collapsing to a single value when they are equal."""
    return after if before == after else f"{before} -> {after}"


def summarize(
    values: np.ndarray, *, savepoint: str, date: str | None, field: str, component: str | None
) -> FieldStats:
    """Reduce an array to the statistics that tell whether it carries information."""
    values = np.squeeze(values)
    as_float = values.astype(np.float64)
    finite = np.isfinite(as_float)
    n_nonfinite = int(as_float.size - finite.sum())
    usable = as_float[finite]
    if usable.size == 0:
        minimum = maximum = mean = nonzero_fraction = float("nan")
    else:
        minimum = float(usable.min())
        maximum = float(usable.max())
        mean = float(usable.mean())
        nonzero_fraction = float((usable != 0.0).mean())
    return FieldStats(
        savepoint=savepoint,
        date=date,
        field=field,
        component=component,
        shape=tuple(values.shape),
        min=minimum,
        max=maximum,
        mean=mean,
        nonzero_fraction=nonzero_fraction,
        n_nonfinite=n_nonfinite,
    )


class ArchiveExplorer:
    """Read-only view on a serialized archive: savepoints, fields, and their statistics."""

    def __init__(
        self, path: pathlib.Path | str, *, prefix: str = DEFAULT_PREFIX, rank: int = 0
    ) -> None:
        self.path = pathlib.Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"No serialized data directory at '{self.path}'.")

        import serialbox  # noqa: PLC0415 [import-outside-top-level]

        self.prefix = prefix
        self.rank = rank
        self._serializer = serialbox.Serializer(
            serialbox.OpenModeKind.Read, str(self.path), f"{prefix}_rank{rank}"
        )
        self._raw = tuple(self._serializer.savepoint_list())
        self.savepoints = tuple(
            SavepointRef(index, savepoint.name, savepoint.metainfo.to_dict().get("date"))
            for index, savepoint in enumerate(self._raw)
        )

    @classmethod
    def for_experiment(
        cls,
        experiment: str,
        *,
        version: int | None = None,
        comm_size: int = 1,
        rank: int = 0,
        prefix: str = DEFAULT_PREFIX,
    ) -> ArchiveExplorer:
        """Open the archive of *experiment*, overriding the version in ``definitions`` if given."""
        description = resolve_experiment(experiment)
        if version is not None:
            description = dataclasses.replace(description, version=version)
        return cls(experiment_data_path(description, comm_size), prefix=prefix, rank=rank)

    def __repr__(self) -> str:
        return (
            f"ArchiveExplorer({str(self.path)!r}, rank={self.rank}, "
            f"savepoints={len(self.savepoints)})"
        )

    @functools.cached_property
    def names(self) -> dict[str, list[SavepointRef]]:
        """Savepoint references grouped by savepoint name, in archive order."""
        grouped: dict[str, list[SavepointRef]] = {}
        for reference in self.savepoints:
            grouped.setdefault(reference.name, []).append(reference)
        return grouped

    def fields(self, reference: SavepointRef) -> tuple[str, ...]:
        return tuple(self._serializer.fields_at_savepoint(self._raw[reference.index]))

    def read(self, reference: SavepointRef, field: str) -> np.ndarray:
        """Raw array of *field*, exactly as serialized (halo entries included)."""
        return self._serializer.read(field, self._raw[reference.index])

    def find(
        self, *, name: str | None = None, date: str | None = None, field: str | None = None
    ) -> list[SavepointRef]:
        """Savepoints matching all given filters; *name* and *date* match on substrings."""
        found = self.savepoints
        if name is not None:
            found = tuple(reference for reference in found if name in reference.name)
        if date is not None:
            found = tuple(
                reference
                for reference in found
                if reference.date is not None and date in reference.date
            )
        if field is not None:
            found = tuple(reference for reference in found if field in self.fields(reference))
        return list(found)

    def stats(
        self, *, name: str | None = None, date: str | None = None, field: str | None = None
    ) -> list[FieldStats]:
        """Statistics of every matching field of every matching savepoint."""
        collected: list[FieldStats] = []
        for reference in self.find(name=name, date=date, field=field):
            for field_name in self.fields(reference):
                if field is not None and field != field_name:
                    continue
                values = self.read(reference, field_name)
                if not np.issubdtype(values.dtype, np.number) and values.dtype != np.bool_:
                    continue
                collected.extend(self._summarize_components(values, reference, field_name))
        return collected

    def diff_stats(self, before: SavepointRef, after: SavepointRef, field: str) -> list[FieldStats]:
        """Statistics of ``after - before``, to see whether a field changes at all.

        The two savepoints may differ in timestamp (does the field evolve?), in name
        (does the process between an init and an exit savepoint do anything?), or both.
        """
        first = self.read(before, field).astype(np.float64)
        second = self.read(after, field).astype(np.float64)
        if first.shape != second.shape:
            raise ValueError(
                f"Shape mismatch for field '{field}': {first.shape} at '{before.label}' "
                f"vs {second.shape} at '{after.label}'."
            )
        labelled = SavepointRef(
            index=after.index,
            name=_transition(before.name, after.name),
            date=_transition(before.date, after.date),
        )
        return self._summarize_components(second - first, labelled, field)

    def _summarize_components(
        self, values: np.ndarray, reference: SavepointRef, field: str
    ) -> list[FieldStats]:
        labels = component_labels(field, values.shape[-1]) if values.ndim > 1 else None
        species = (
            [(values, None)]
            if labels is None
            else [(values[..., index], label) for index, label in enumerate(labels)]
        )
        return [
            summarize(
                slice_, savepoint=reference.name, date=reference.date, field=field, component=label
            )
            for slice_, label in species
        ]


# -- text output --


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render *rows* as a left-aligned fixed-width table.

    Not 'rich.table': it truncates to the terminal width, and savepoint and field names
    are the whole point of this output.
    """
    if not rows:
        return "(no rows)"
    widths = [
        max(len(header), *(len(row[column]) for row in rows))
        for column, header in enumerate(headers)
    ]
    lines = [
        "  ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    )
    return "\n".join(lines)


def _number(value: float) -> str:
    return f"{value:.6g}"


def format_stats(collected: Sequence[FieldStats]) -> str:
    rows = [
        [
            stats.savepoint,
            stats.date or "",
            stats.full_name,
            "x".join(str(size) for size in stats.shape),
            _number(stats.min),
            _number(stats.max),
            _number(stats.mean),
            f"{stats.nonzero_fraction:.3f}",
            str(stats.n_nonfinite) if stats.n_nonfinite else "",
            "ALL ZEROS" if stats.all_zero else "",
        ]
        for stats in collected
    ]
    headers = ["savepoint", "date", "field", "shape", "min", "max", "mean", "nonzero", "nonfin", ""]
    return format_table(headers, rows)


# -- CLI --

ExperimentOption = Annotated[
    str | None,
    typer.Option("--experiment", "-e", help="Experiment name, e.g. 'exclaim_ape_aesPhys'."),
]
VersionOption = Annotated[
    int | None,
    typer.Option("--version", "-v", help="Archive version; defaults to the one in 'definitions'."),
]
CommSizeOption = Annotated[int, typer.Option("--comm-size", "-c", help="Number of MPI ranks.")]
RankOption = Annotated[int, typer.Option("--rank", "-r", help="Rank whose archive is read.")]
PathOption = Annotated[
    pathlib.Path | None,
    typer.Option("--path", "-p", help="Path to a 'ser_data' directory, instead of '--experiment'."),
]
NameOption = Annotated[
    str | None, typer.Option("--name", "-n", help="Only savepoints whose name contains this.")
]
DateOption = Annotated[
    str | None, typer.Option("--date", "-d", help="Only savepoints whose date contains this.")
]
FieldOption = Annotated[str | None, typer.Option("--field", "-f", help="Only this field.")]


def open_archive(
    experiment: str | None,
    version: int | None,
    comm_size: int,
    rank: int,
    path: pathlib.Path | None,
) -> ArchiveExplorer:
    if (experiment is None) == (path is None):
        raise typer.BadParameter("Pass either '--experiment' or '--path', but not both.")
    try:
        if experiment is None:
            return ArchiveExplorer(path, rank=rank)
        return ArchiveExplorer.for_experiment(
            experiment, version=version, comm_size=comm_size, rank=rank
        )
    except FileNotFoundError as error:
        raise typer.BadParameter(str(error)) from error


@cli.command()
def savepoints(
    *,
    experiment: ExperimentOption = None,
    version: VersionOption = None,
    comm_size: CommSizeOption = 1,
    rank: RankOption = 0,
    path: PathOption = None,
    name: NameOption = None,
) -> None:
    """List the savepoints of an archive, grouped by name."""
    archive = open_archive(experiment, version, comm_size, rank, path)
    print(f"{archive.path} ({len(archive.savepoints)} savepoints)")
    rows = []
    for savepoint_name, references in archive.names.items():
        if name is not None and name not in savepoint_name:
            continue
        dates = [reference.date for reference in references if reference.date]
        fields = archive.fields(references[0])
        listed = ", ".join(fields[:MAX_LISTED_FIELDS])
        if len(fields) > MAX_LISTED_FIELDS:
            listed += f", ... (+{len(fields) - MAX_LISTED_FIELDS})"
        rows.append(
            [
                savepoint_name,
                str(len(references)),
                dates[0] if dates else "",
                dates[-1] if len(dates) > 1 else "",
                listed,
            ]
        )
    print(format_table(["savepoint", "count", "first date", "last date", "fields"], rows))


@cli.command()
def stats(
    *,
    experiment: ExperimentOption = None,
    version: VersionOption = None,
    comm_size: CommSizeOption = 1,
    rank: RankOption = 0,
    path: PathOption = None,
    name: NameOption = None,
    date: DateOption = None,
    field: FieldOption = None,
) -> None:
    """Summarize the fields of the matching savepoints.

    Without filters every field of every savepoint is read, which is slow on a
    full archive; narrow it down with '--name' or '--field'.
    """
    archive = open_archive(experiment, version, comm_size, rank, path)
    collected = archive.stats(name=name, date=date, field=field)
    if not collected:
        raise typer.BadParameter("No savepoint matches the given filters.")
    print(archive.path)
    print(format_stats(collected))
    empty = [entry for entry in collected if entry.all_zero]
    if empty:
        print(f"\n{len(empty)} of {len(collected)} entries are all-zero.")


@cli.command()
def diff(
    *,
    field: Annotated[str, typer.Option("--field", "-f", help="Field to difference.")],
    experiment: ExperimentOption = None,
    version: VersionOption = None,
    comm_size: CommSizeOption = 1,
    rank: RankOption = 0,
    path: PathOption = None,
    name: NameOption = None,
    against: Annotated[
        str | None,
        typer.Option("--against", "-a", help="Diff against this savepoint at the same dates."),
    ] = None,
    from_date: Annotated[
        str | None, typer.Option("--from", help="Start date; default is all consecutive pairs.")
    ] = None,
    to_date: Annotated[str | None, typer.Option("--to", help="End date.")] = None,
) -> None:
    """Summarize how a field changes between two savepoints.

    By default every consecutive pair of timestamps of one savepoint is compared,
    which shows whether the field evolves. With '--from'/'--to' only those two
    timestamps are. With '--against' the comparison is against another savepoint at
    the same dates instead, which shows whether the process in between changes the
    field at all (for instance an init against its exit).
    """
    archive = open_archive(experiment, version, comm_size, rank, path)
    references = _one_savepoint(archive.find(name=name, field=field), field, "--name")

    if against is not None:
        others = _one_savepoint(archive.find(name=against, field=field), field, "--against")
        by_date = {reference.date: reference for reference in others}
        pairs = [
            (reference, by_date[reference.date])
            for reference in references
            if reference.date in by_date
        ]
        if not pairs:
            raise typer.BadParameter(
                f"'{references[0].name}' and '{others[0].name}' share no date."
            )
    elif from_date is None and to_date is None:
        if len(references) < 2:
            raise typer.BadParameter(
                f"Only one savepoint holds field '{field}'; pass '--against' to compare it "
                "with another savepoint."
            )
        pairs = list(itertools.pairwise(references))
    else:
        pairs = [(_at_date(references, from_date), _at_date(references, to_date))]

    collected = [
        entry for before, after in pairs for entry in archive.diff_stats(before, after, field)
    ]
    print(f"{archive.path}: differences of '{field}'")
    print(format_stats(collected))
    unchanged = [entry for entry in collected if entry.all_zero]
    if unchanged:
        print(f"\n{len(unchanged)} of {len(collected)} differences are identically zero.")


def _one_savepoint(
    references: Sequence[SavepointRef], field: str, option: str
) -> list[SavepointRef]:
    """Check that *references* all belong to a single savepoint that holds *field*."""
    if not references:
        raise typer.BadParameter(f"No savepoint matching '{option}' holds field '{field}'.")
    names = {reference.name for reference in references}
    if len(names) > 1:
        raise typer.BadParameter(
            f"Field '{field}' occurs in several savepoints, narrow '{option}' down: "
            f"{', '.join(sorted(names))}."
        )
    return list(references)


def _at_date(references: Sequence[SavepointRef], date: str | None) -> SavepointRef:
    if date is None:
        raise typer.BadParameter("Pass both '--from' and '--to', or neither.")
    matching = [reference for reference in references if reference.date == date]
    if len(matching) != 1:
        available = ", ".join(sorted({r.date for r in references if r.date}))
        raise typer.BadParameter(f"No unique savepoint at date '{date}'. Available: {available}.")
    return matching[0]


@cli.command()
def shell(
    *,
    experiment: ExperimentOption = None,
    version: VersionOption = None,
    comm_size: CommSizeOption = 1,
    rank: RankOption = 0,
    path: PathOption = None,
) -> None:
    """Open an interactive interpreter with the archive bound to 'e'."""
    archive = open_archive(experiment, version, comm_size, rank, path)
    namespace = {"e": archive, "np": np, "ArchiveExplorer": ArchiveExplorer}
    banner = (
        f"{archive!r}\n"
        "  e.names, e.savepoints, e.find(name=..), e.fields(ref), e.read(ref, field)\n"
        "  e.stats(name=.., field=..), e.diff_stats(before, after, field)"
    )
    print(banner)
    try:
        import IPython  # noqa: PLC0415 [import-outside-top-level]
    except ImportError:
        import code  # noqa: PLC0415 [import-outside-top-level]

        code.interact(banner="", local=namespace)
    else:
        IPython.start_ipython(argv=[], user_ns=namespace, display_banner=False)


if __name__ == "__main__":
    cli()
