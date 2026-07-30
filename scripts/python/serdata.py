#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect and compare serialized ICON test data archives."""

from __future__ import annotations

import collections
import pathlib
import sys
from typing import Annotated

import typer

from icon4py.model.testing.serialized_data import (
    SAVEPOINT_CLASSES,
    backfill_archive_metadata,
    compare_archives,
    fingerprint_archive,
    render_diff_report,
    summarise_differences,
)


cli = typer.Typer(no_args_is_help=True, name="serdata", help=__doc__)


@cli.command()
def fingerprint(
    archive: Annotated[pathlib.Path, typer.Argument(help="Extracted archive directory.")],
) -> None:
    """Summarise the per-field digests of an extracted archive."""
    result = fingerprint_archive(archive)
    counts: collections.Counter = collections.Counter(
        result.classes.get(key[1], "evolving") for key in result.records
    )
    print(f"{archive.name}: {result.ranks} rank(s), {len(result.records)} records")
    for name in SAVEPOINT_CLASSES:
        print(f"  {name:<14} {counts[name]}")
    if result.unclassified:
        print(f"  UNCLASSIFIED savepoints: {', '.join(result.unclassified)}")


@cli.command()
def diff(
    new: Annotated[pathlib.Path, typer.Argument(help="Freshly generated archive directory.")],
    baseline: Annotated[
        pathlib.Path | None,
        typer.Option("--baseline", help="Archive to compare against; the previous version."),
    ] = None,
    report: Annotated[
        pathlib.Path | None, typer.Option("--report", help="Also write the report here.")
    ] = None,
) -> None:
    """Report what changed between an archive and its predecessor."""
    comparison = compare_archives(baseline, new)
    rendered = render_diff_report(comparison)
    print(rendered)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(rendered + "\n")


@cli.command()
def backfill(
    testdata_root: Annotated[
        pathlib.Path, typer.Argument(help="Directory holding the extracted archives.")
    ],
) -> None:
    """Reconstruct metadata for archives generated before it existed."""
    written = backfill_archive_metadata(testdata_root)
    for path in written:
        print(f"  wrote {path}")
    print(f"{len(written)} archive(s) backfilled")


if __name__ == "__main__":
    sys.exit(cli())


@cli.command()
def inspect(
    new: Annotated[pathlib.Path, typer.Argument(help="Freshly generated archive directory.")],
    baseline: Annotated[
        pathlib.Path, typer.Option("--baseline", help="Archive to compare against.")
    ],
    limit: Annotated[int, typer.Option(help="Differing entries to show per field.")] = 5,
) -> None:
    """Show the values behind every changed guarded record.

    A digest says that a field changed, not how. This reads the two arrays so that a
    static-class hit can be triaged: a genuine change looks nothing like uninitialised
    padding, but only the values tell them apart.
    """
    import serialbox  # noqa: PLC0415 [import-outside-top-level]

    def read(archive_dir: pathlib.Path, rank: int, savepoint: str, ordinal: int, field: str):
        serializer = serialbox.Serializer(
            serialbox.OpenModeKind.Read, str(archive_dir / "ser_data"), f"icon_pydycore_rank{rank}"
        )
        matching = [p for p in serializer.savepoint_list() if p.name == savepoint]
        return serializer.read(field, matching[ordinal])

    comparison = compare_archives(baseline, new)
    for name in ("static", "initial-state"):
        for key in comparison.fingerprints.per_class[name].changed:
            rank, savepoint, ordinal, field = key
            summary = summarise_differences(
                read(baseline, rank, savepoint, ordinal, field),
                read(new, rank, savepoint, ordinal, field),
                limit=limit,
            )
            print(
                f"{savepoint}#{ordinal} {field} (rank {rank}): "
                f"{summary.count} of {summary.total} entries differ"
            )
            for position, old_value, new_value in summary.samples:
                print(f"    {position}: {old_value} -> {new_value}")
