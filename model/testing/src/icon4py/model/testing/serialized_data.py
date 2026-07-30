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
from collections.abc import Iterable, Iterator
from typing import Any, Final, Literal


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


def _read_rank(
    meta_path: pathlib.Path, archive_path: pathlib.Path, rank: int
) -> tuple[dict[RecordKey, str], dict[str, SavepointClass], list[tuple[str, dict]]]:
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

    def of_class(keys: Iterable[RecordKey], name: SavepointClass) -> list[RecordKey]:
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


# ---------------------------------------------------------------------------
# Namelists
# ---------------------------------------------------------------------------

# ICON's post-read namelist dump: every variable with its effective value, defaults
# included. The input namelists only carry what the experiment sets explicitly, so this
# is the only archived artifact in which an upstream change to a default is visible.
NAMELIST_DUMP_FNAME: Final = "NAMELIST_ICON_output_atm.json"


@dataclasses.dataclass(frozen=True)
class NamelistDiff:
    changed: list[tuple[str, str, object, object]]
    added: list[tuple[str, str]]
    removed: list[tuple[str, str]]


def read_namelist(archive_dir: pathlib.Path) -> dict:
    """Read the resolved namelist dump of an archive, or nothing if it has none."""
    path = archive_dir / NAMELIST_DUMP_FNAME
    return json.loads(path.read_text()) if path.is_file() else {}


def diff_namelists(old: dict, new: dict) -> NamelistDiff:
    """Compare two resolved namelist dumps variable by variable."""
    changed: list[tuple[str, str, object, object]] = []
    added: list[tuple[str, str]] = []
    removed: list[tuple[str, str]] = []

    for section in sorted(old.keys() | new.keys()):
        old_section = old.get(section, {})
        new_section = new.get(section, {})
        for key in sorted(old_section.keys() | new_section.keys()):
            if key not in old_section:
                added.append((section, key))
            elif key not in new_section:
                removed.append((section, key))
            elif old_section[key] != new_section[key]:
                changed.append((section, key, old_section[key], new_section[key]))

    return NamelistDiff(changed=changed, added=added, removed=removed)


# ---------------------------------------------------------------------------
# Comparing two archives
# ---------------------------------------------------------------------------

ARCHIVE_METADATA_FNAME: Final = "archive_metadata.json"
ARCHIVE_METADATA_SCHEMA: Final = "icon4py-archive-metadata/1"

_MAX_RENDERED_VALUE = 60
_MAX_RENDERED_ELEMENTS = 3


def read_archive_provenance(archive_dir: pathlib.Path) -> dict:
    """Read an archive's provenance, from its metadata file or from the ICON log.

    Archives generated before the metadata file existed still carry the log, so the
    fallback is what lets a fresh archive be compared against an older one.
    """
    metadata_path = archive_dir / ARCHIVE_METADATA_FNAME
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text()).get("provenance", {})
    try:
        return {"icon": read_icon_banner_from_archive(archive_dir)}
    except BannerParseError as error:
        return {"error": str(error)}


@dataclasses.dataclass(frozen=True)
class ArchiveComparison:
    """Everything that changed between two versions of one experiment's archive."""

    experiment: str
    old_label: str | None
    new_label: str
    old_provenance: dict
    new_provenance: dict
    namelists: NamelistDiff
    fingerprints: FingerprintDiff
    unclassified: list[str]

    @property
    def verdict(self) -> str:
        """'OK', 'REVIEW' if something guarded moved, 'UNVERIFIED' without a baseline.

        Records appearing in a guarded class are not by themselves cause for review:
        that is what adding instrumentation looks like. Records changing or vanishing
        are.
        """
        if self.old_label is None:
            return "UNVERIFIED"
        if self.unclassified:
            return "REVIEW"
        for name in ("static", "initial-state"):
            guarded = self.fingerprints.per_class[name]
            if guarded.changed or guarded.removed:
                return "REVIEW"
        return "OK"


def compare_archives(
    old_dir: pathlib.Path | None, new_dir: pathlib.Path, *, experiment: str | None = None
) -> ArchiveComparison:
    """Compare a freshly generated archive against its predecessor.

    The baseline is fingerprinted on the spot from its own 'ser_data', so a comparison
    never depends on a previous run having written anything.
    """
    new = fingerprint_archive(new_dir)
    old = (
        fingerprint_archive(old_dir)
        if old_dir is not None
        else Fingerprint(records={}, classes={}, ranks=0, unclassified=[])
    )

    return ArchiveComparison(
        experiment=experiment or new_dir.name,
        old_label=old_dir.name if old_dir is not None else None,
        new_label=new_dir.name,
        old_provenance=read_archive_provenance(old_dir) if old_dir is not None else {},
        new_provenance=read_archive_provenance(new_dir),
        namelists=diff_namelists(
            read_namelist(old_dir) if old_dir is not None else {}, read_namelist(new_dir)
        ),
        fingerprints=diff_fingerprints(old, new),
        unclassified=new.unclassified,
    )


def _abbreviate(value: object) -> str:
    # Fortran pads strings to a fixed width, so the padding has to go before the value
    # is truncated, and quoting is what makes an emptied string visible.
    text = repr(value.strip()) if isinstance(value, str) else str(value)
    return text if len(text) <= _MAX_RENDERED_VALUE else text[: _MAX_RENDERED_VALUE - 3] + "..."


def render_value_change(old: object, new: object) -> str:
    """Describe how one namelist value changed.

    Array values are long and mostly identical, so naming the differing positions is
    the only rendering that tells the reader anything.
    """
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        differing = [i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b]
        if differing:
            shown = differing[:_MAX_RENDERED_ELEMENTS]
            rendered = ", ".join(
                f"[{i}] {_abbreviate(old[i])} -> {_abbreviate(new[i])}" for i in shown
            )
            omitted = len(differing) - len(shown)
            return rendered + (f", and {omitted} more" if omitted else "")
    return f"{_abbreviate(old)} -> {_abbreviate(new)}"


def _render_provenance(comparison: ArchiveComparison) -> str:
    def sha(provenance: dict) -> str:
        return (provenance.get("icon", {}).get("sha") or "unknown")[:10]

    if comparison.old_label is None:
        return f"provenance : icon {sha(comparison.new_provenance)}"
    return f"provenance : icon {sha(comparison.old_provenance)} -> {sha(comparison.new_provenance)}"


def _render_savepoints(keys: list[RecordKey]) -> str:
    return ", ".join(sorted({key[1] for key in keys}))


def render_diff_report(comparison: ArchiveComparison) -> str:
    """Render the comparison as the few lines a human reads before publishing."""
    old_label = comparison.old_label or "none"
    lines = [
        f"{comparison.experiment}  {old_label} -> {comparison.new_label}   "
        f"VERDICT: {comparison.verdict}",
    ]
    if comparison.old_label is None:
        lines.append("BASELINE: none -- no comparison performed")
    lines.append(_render_provenance(comparison))

    namelists = comparison.namelists
    lines.append(
        f"namelists  : {len(namelists.changed)} changed, {len(namelists.added)} added, "
        f"{len(namelists.removed)} removed"
    )
    for section, key, old_value, new_value in namelists.changed:
        lines.append(f"             {section}.{key}  {render_value_change(old_value, new_value)}")

    fingerprints = comparison.fingerprints
    lines.append(
        f"structure  : savepoints +{len(fingerprints.savepoints_added)} "
        f"-{len(fingerprints.savepoints_removed)} | fields "
        f"+{len(fingerprints.fields_added)} -{len(fingerprints.fields_removed)}"
    )
    for name in SAVEPOINT_CLASSES:
        class_diff = fingerprints.per_class[name]
        lines.append(
            f"{name.upper():<14} {class_diff.unchanged} unchanged, "
            f"{len(class_diff.changed)} changed, {len(class_diff.added)} added, "
            f"{len(class_diff.removed)} removed"
        )
        # Guarded classes are small enough to name field by field; the trajectory is not.
        if name != "evolving":
            for label, keys in (("changed", class_diff.changed), ("removed", class_diff.removed)):
                for savepoint in sorted({key[1] for key in keys}):
                    fields = sorted({key[3] for key in keys if key[1] == savepoint})
                    lines.append(f"  {label} {savepoint:<22} {', '.join(fields)}")
            if class_diff.added:
                lines.append(f"  added savepoints: {_render_savepoints(class_diff.added)}")

    if comparison.unclassified:
        lines.append(f"UNCLASSIFIED savepoints: {', '.join(comparison.unclassified)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backfilling archives generated before this metadata existed
# ---------------------------------------------------------------------------

# Inverse of 'icon4py.model.testing.datatest_utils.get_ranked_experiment_name_with_version'.
# Kept as a pattern rather than an import so that this module stays usable without an
# icon4py environment; 'scripts/tests/python/test_run_serialization.py' pins the two
# together.
_ARCHIVE_DIRNAME = re.compile(r"^mpitask(?P<comm_size>\d+)_(?P<experiment>.+)_v(?P<version>\d+)$")


def parse_archive_dirname(name: str) -> tuple[int, str, int] | None:
    """Recover '(comm_size, experiment, version)' from an archive directory name."""
    match = _ARCHIVE_DIRNAME.match(name)
    if match is None:
        return None
    return int(match["comm_size"]), match["experiment"], int(match["version"])


def backfill_archive_metadata(testdata_root: pathlib.Path) -> list[pathlib.Path]:
    """Write metadata for already extracted archives, from the logs they carry.

    This makes archives that predate the metadata file usable as a diff baseline
    without regenerating or re-publishing anything. Archives whose log cannot be read
    are reported and skipped, so one bad archive does not stop the rest.
    """
    written = []
    for archive_dir in sorted(p for p in testdata_root.iterdir() if p.is_dir()):
        parsed = parse_archive_dirname(archive_dir.name)
        if parsed is None:
            continue
        metadata_path = archive_dir / ARCHIVE_METADATA_FNAME
        if metadata_path.exists():
            continue

        comm_size, experiment, version = parsed
        try:
            icon = read_icon_banner_from_archive(archive_dir)
        except BannerParseError as error:
            print(f"  skipped {archive_dir.name}: {error}")
            continue

        metadata = {
            "schema": ARCHIVE_METADATA_SCHEMA,
            # 'backfilled' marks this as reconstructed after the fact: the runtime
            # section a generation run would record cannot be recovered from the log.
            "archive": {
                "experiment": experiment,
                "version": version,
                "comm_size": comm_size,
                "backfilled": True,
            },
            "provenance": {"icon": icon},
        }
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=4)
        written.append(metadata_path)

    return written


# ---------------------------------------------------------------------------
# Triaging a changed record
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DifferenceSummary:
    """Where two versions of one field differ, and by how much."""

    count: int
    total: int
    samples: list[tuple[tuple[int, ...], object, object]]


def summarise_differences(old: Any, new: Any, *, limit: int = 10) -> DifferenceSummary:
    """Locate the entries in which two same-shaped arrays differ.

    Kept free of any array library so that it can be tested without data files; the
    command below feeds it whatever Serialbox returns.
    """

    def walk(a: Any, b: Any, prefix: tuple[int, ...]) -> Iterator[tuple[tuple[int, ...], Any, Any]]:
        if isinstance(a, (list, tuple)) or hasattr(a, "__len__"):
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                yield from walk(x, y, (*prefix, i))
        elif a != b:
            yield prefix, a, b

    count = 0
    samples: list[tuple[tuple[int, ...], object, object]] = []
    for position, old_value, new_value in walk(old, new, ()):
        count += 1
        if len(samples) < limit:
            samples.append((position, old_value, new_value))

    # 'walk' only yields differing entries, so the size is counted separately.
    return DifferenceSummary(count=count, total=_count_entries(old), samples=samples)


def _count_entries(array: Any) -> int:
    if isinstance(array, (list, tuple)) or hasattr(array, "__len__"):
        return sum(_count_entries(item) for item in array)
    return 1


# ---------------------------------------------------------------------------
# Lockfile
# ---------------------------------------------------------------------------

TESTDATA_LOCK_SCHEMA: Final = "icon4py-testdata-lock/1"

# Digests only ever have to differ from one another, not to be reproducible outside
# Serialbox, so the lockfile stores a prefix and stays small enough to read in a diff.
_LOCKED_DIGEST_LENGTH: Final = 16

# Classes whose records are pinned. The trajectory is deliberately absent: it is
# expected to move, and locking it would make every regeneration a conflict.
LOCKED_CLASSES: Final[tuple[SavepointClass, ...]] = ("static", "initial-state")


def _lock_key(key: RecordKey) -> str:
    rank, savepoint, ordinal, field = key
    return f"{savepoint}#{ordinal}/{field}@rank{rank}"


def build_lock_entry(fingerprint: Fingerprint, *, comm_size: int) -> dict:
    """Pin the guarded records of one archive."""
    records = {
        _lock_key(key): digest[:_LOCKED_DIGEST_LENGTH]
        for key, digest in fingerprint.records.items()
        if fingerprint.classes.get(key[1], "evolving") in LOCKED_CLASSES
    }
    return {"comm_size": comm_size, "ranks": fingerprint.ranks, "records": records}


def verify_against_lock(lock_entry: dict, fingerprint: Fingerprint) -> list[str]:
    """Report every pinned record that changed or disappeared.

    Records the lock does not know about are ignored: adding instrumentation adds
    records, and that is not a violation of what was blessed.
    """
    current = {_lock_key(key): digest for key, digest in fingerprint.records.items()}
    problems = []
    for key, locked in sorted(lock_entry["records"].items()):
        if key not in current:
            problems.append(f"{key}: missing from the archive")
        elif current[key][:_LOCKED_DIGEST_LENGTH] != locked:
            problems.append(f"{key}: locked {locked}, found {current[key][:_LOCKED_DIGEST_LENGTH]}")
    return problems
