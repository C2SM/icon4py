#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Metadata and drift detection for serialized ICON test data.

Serialbox already stores a SHA256 for every field it writes, and ICON already prints
its git revision at startup into the slurm log that ships inside every archive. This
module turns both into machine-readable metadata so that a regeneration can be diffed
against its predecessor instead of being taken on trust.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import pathlib
import re
import subprocess
from collections.abc import Iterator
from typing import Final, Literal


# Top-level entries of the banner printed by 'mo_util_vcs::show_version'. They carry
# exactly one leading space; nested components are indented further.
_BANNER_KEYS: dict[str, str] = {
    "executable": "executable",
    "date": "build_date",
    "time": "build_time",
    "user": "user",
    "host": "host",
    "version": "version",
    "revision": "describe",
    "repository": "repository",
    "local branch": "branch",
}

# Banner sections and the group each one contributes to.
_BANNER_SECTIONS: dict[str, str] = {
    "model components": "externals",
    "application libraries": "externals",
    "infrastructure and support libraries": "externals",
    "other libraries": "toolchain",
    "compilers": "toolchain",
}

_TOP_LEVEL_ENTRY = re.compile(r"^ (?P<key>[^ :][^:]*): ?(?P<value>.*)$")
_NESTED_ENTRY = re.compile(r"^(?P<indent> {2,})(?P<key>[^:]+): ?(?P<value>.*)$")
_DESCRIBE_SHA = re.compile(r"-g(?P<sha>[0-9a-f]{7,40})$")


class BannerParseError(RuntimeError):
    """Raised when an ICON log does not contain a usable version banner."""


def _iter_banner_entries(text: str) -> Iterator[tuple[str, int, str, str]]:
    """Yield '(group, indent, key, value)' for the banner region of a log.

    The group is 'top' for ICON's own entries and the name of the collection a nested
    component belongs to. Iteration stops at the first line that is regular model output.
    """
    group = "externals"
    started = False

    for line in text.splitlines():
        top_level = _TOP_LEVEL_ENTRY.match(line)
        if top_level:
            key = top_level["key"].strip()
            if key in _BANNER_KEYS:
                started = True
                yield "top", 1, key, top_level["value"].strip()
            elif not started:
                continue
            elif key in _BANNER_SECTIONS:
                group = _BANNER_SECTIONS[key]
            else:
                return
            continue

        nested = _NESTED_ENTRY.match(line)
        if started and nested:
            yield (
                group,
                len(nested["indent"]),
                nested["key"].strip().lower(),
                nested["value"].strip(),
            )


def parse_icon_log_banner(text: str) -> dict:
    """Extract ICON's startup version banner from a slurm log.

    Returns the top-level identity (revision, repository, branch, build host and time)
    plus two flat mappings: 'externals' for the bundled model components and libraries,
    and 'toolchain' for compilers and system libraries.
    """
    banner: dict = {"externals": {}, "toolchain": {}}
    # A component may announce itself on a bare 'name:' line and give its revision on a
    # deeper-indented line below.
    pending_name: str | None = None
    pending_indent = 0

    for group, indent, key, value in _iter_banner_entries(text):
        if group == "top":
            banner[_BANNER_KEYS[key]] = value
            pending_name = None
        elif pending_name is not None and indent > pending_indent:
            if key == "revision":
                banner[group][pending_name] = value
        elif value:
            pending_name = None
            banner[group][key] = value
        else:
            pending_name, pending_indent = key, indent

    if not set(banner) - {"externals", "toolchain"}:
        raise BannerParseError("No ICON version banner found in the log.")
    if "describe" not in banner:
        raise BannerParseError("ICON version banner has no 'revision' entry.")

    sha = _DESCRIBE_SHA.search(banner["describe"])
    if sha is None:
        raise BannerParseError(f"Cannot extract a commit sha from revision '{banner['describe']}'.")
    banner["sha"] = sha["sha"]
    if "host" in banner:
        banner["host"] = banner["host"].split(" ", 1)[0]

    return banner


def read_icon_banner_from_archive(archive_dir: pathlib.Path) -> dict:
    """Parse the ICON version banner from the slurm log shipped inside an archive."""
    logs = sorted(archive_dir.glob("LOG.*.o"))
    if not logs:
        raise BannerParseError(f"No ICON log found in '{archive_dir}'.")
    if len(logs) > 1:
        names = ", ".join(log.name for log in logs)
        raise BannerParseError(
            f"Found several ICON logs in '{archive_dir}', cannot tell which run produced "
            f"the data: {names}."
        )
    banner = parse_icon_log_banner(logs[0].read_text(errors="replace"))
    banner["source"] = logs[0].name
    return banner


def harvest_git(repo: pathlib.Path) -> dict:
    """Record the git state of a working tree, or an empty mapping if there is none."""

    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {}
    status = run("status", "--porcelain")
    return {
        "sha": sha,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


# ---------------------------------------------------------------------------
# Savepoint classification
# ---------------------------------------------------------------------------

SavepointClass = Literal["static", "initial-state", "evolving"]

# The authoritative order in which classes are reported, strictest first.
SAVEPOINT_CLASSES: Final[tuple[SavepointClass, ...]] = ("static", "initial-state", "evolving")

# Savepoints written once at startup, describing time-invariant model state. A change
# here is a semantic change in ICON, not the trajectory moving, so it has to be
# justified. The list is explicit on purpose: a heuristic that silently absorbs a new
# savepoint would silently stop guarding it.
STATIC_SAVEPOINTS: Final = frozenset(
    {
        "icon-grid",
        "interpolation-state",
        "metric-state",
        "smooth-topo-savepoint",
        "tmx-init",
    }
)

# Metainfo keys that mark a savepoint as belonging to the time loop.
_TIME_KEYS: Final = frozenset({"date", "dyn_timestep"})


def classify_savepoint(name: str, meta_info: dict) -> SavepointClass:
    """Assign a savepoint to the class that decides how strictly it is compared."""
    # 'initial-state' is tested first: those savepoints carry neither a date nor a
    # dyn_timestep, so a "written once" rule alone would swallow them into 'static'.
    if meta_info.get("location") == "initial-state":
        return "initial-state"
    if name in STATIC_SAVEPOINTS:
        return "static"
    return "evolving"


def unclassified_savepoints(savepoints: list[tuple[str, dict]]) -> list[str]:
    """Report savepoints that look time-invariant but are not classified as such.

    These are new savepoints that a human has to place explicitly, rather than let them
    default into the unguarded 'evolving' class.
    """
    flagged = {
        name
        for name, meta_info in savepoints
        if classify_savepoint(name, meta_info) == "evolving" and not (_TIME_KEYS & set(meta_info))
    }
    return sorted(flagged)


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

# (rank, savepoint name, occurrence ordinal, field name)
RecordKey = tuple[int, str, int, str]

_RANK_FROM_METADATA = re.compile(r"MetaData-.*_rank(?P<rank>\d+)\.json$")


@dataclasses.dataclass
class Fingerprint:
    """Per-field digests of one archive, keyed per rank."""

    records: dict[RecordKey, str]
    classes: dict[str, SavepointClass]
    ranks: int
    # Savepoints that look time-invariant but are not in 'STATIC_SAVEPOINTS'. They are
    # compared as 'evolving', which means they are not guarded, so they have to be
    # classified by a human before the next regeneration.
    unclassified: list[str]


def _read_rank(meta_path: pathlib.Path, archive_path: pathlib.Path, rank: int) -> tuple:
    meta = json.loads(meta_path.read_text())
    archive = json.loads(archive_path.read_text())

    savepoints = meta["savepoint_vector"]["savepoints"]
    fields_per_savepoint = meta["savepoint_vector"]["fields_per_savepoint"]
    if len(savepoints) != len(fields_per_savepoint):
        raise ValueError(f"Inconsistent savepoint vector in '{meta_path}'.")

    # 'fields_table[field]' is an ordered list of [byte offset, digest]; the occurrence
    # index used by 'fields_per_savepoint' is the list position, and element 0 is an
    # offset into the .dat file, never an identifier to look up.
    digests = {
        field: [entry[1] for entry in entries] for field, entries in archive["fields_table"].items()
    }

    records: dict[RecordKey, str] = {}
    classes: dict[str, SavepointClass] = {}
    seen: collections.Counter = collections.Counter()
    described: list[tuple[str, dict]] = []

    for savepoint, fields in zip(savepoints, fields_per_savepoint, strict=True):
        name = savepoint["name"]
        ordinal = seen[name]
        seen[name] += 1
        meta_info = {
            key: value["value"] for key, value in (savepoint.get("meta_info") or {}).items()
        }
        described.append((name, meta_info))
        classes[name] = classify_savepoint(name, meta_info)
        # 'field_map' is a superset of what was actually written; only the fields listed
        # per savepoint are guaranteed to have a digest.
        for field, occurrence in (fields[name] or {}).items():
            # Digests are Serialbox's own 32 byte tokens, printed byte by byte with
            # leading zeros stripped. They are 54-64 characters long, are not
            # interchangeable with 'hashlib.sha256', and must never be padded.
            records[(rank, name, ordinal, field)] = digests[field][occurrence]

    return records, classes, described


def fingerprint_archive(archive_dir: pathlib.Path) -> Fingerprint:
    """Fingerprint every field of every savepoint of an extracted archive.

    Serialbox already stores a digest per written field, so this reads two small JSON
    files per rank rather than the data itself.
    """
    ser_data = archive_dir / "ser_data"
    meta_paths = sorted(ser_data.glob("MetaData-*_rank*.json"))
    if not meta_paths:
        raise FileNotFoundError(f"No Serialbox metadata found in '{ser_data}'.")

    records: dict[RecordKey, str] = {}
    classes: dict[str, SavepointClass] = {}
    described: list[tuple[str, dict]] = []
    for meta_path in meta_paths:
        match = _RANK_FROM_METADATA.search(meta_path.name)
        if match is None:
            raise ValueError(f"Cannot determine the rank of '{meta_path}'.")
        rank = int(match["rank"])
        archive_path = meta_path.with_name(
            meta_path.name.replace("MetaData-", "ArchiveMetaData-", 1)
        )
        rank_records, rank_classes, rank_savepoints = _read_rank(meta_path, archive_path, rank)
        records.update(rank_records)
        classes.update(rank_classes)
        described.extend(rank_savepoints)

    return Fingerprint(
        records=records,
        classes=classes,
        ranks=len(meta_paths),
        unclassified=unclassified_savepoints(described),
    )


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ClassDiff:
    """How one savepoint class changed between two archives.

    Changed, added and removed records are named because the report has to point at
    them; unchanged records are only ever counted.
    """

    unchanged: int
    changed: list[RecordKey]
    added: list[RecordKey]
    removed: list[RecordKey]


@dataclasses.dataclass(frozen=True)
class FingerprintDiff:
    per_class: dict[SavepointClass, ClassDiff]
    savepoints_added: list[str]
    savepoints_removed: list[str]
    fields_added: list[str]
    fields_removed: list[str]


def diff_fingerprints(old: Fingerprint, new: Fingerprint) -> FingerprintDiff:
    """Compare two archives record by record, split by savepoint class."""
    classes = {**old.classes, **new.classes}

    def class_of(key: RecordKey) -> SavepointClass:
        return classes.get(key[1], "evolving")

    common = old.records.keys() & new.records.keys()
    only_new = new.records.keys() - old.records.keys()
    only_old = old.records.keys() - new.records.keys()

    def of_class(keys, name: SavepointClass) -> list[RecordKey]:
        return sorted(key for key in keys if class_of(key) == name)

    per_class: dict[SavepointClass, ClassDiff] = {}
    for name in SAVEPOINT_CLASSES:
        same = of_class((key for key in common if old.records[key] == new.records[key]), name)
        per_class[name] = ClassDiff(
            unchanged=len(same),
            changed=of_class((key for key in common if old.records[key] != new.records[key]), name),
            added=of_class(only_new, name),
            removed=of_class(only_old, name),
        )

    old_savepoints = {key[1] for key in old.records}
    new_savepoints = {key[1] for key in new.records}
    old_fields = {key[3] for key in old.records}
    new_fields = {key[3] for key in new.records}

    return FingerprintDiff(
        per_class=per_class,
        savepoints_added=sorted(new_savepoints - old_savepoints),
        savepoints_removed=sorted(old_savepoints - new_savepoints),
        fields_added=sorted(new_fields - old_fields),
        fields_removed=sorted(old_fields - new_fields),
    )
