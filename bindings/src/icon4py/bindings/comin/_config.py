# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
ICON's configuration, read from its namelist output 'NAMELIST_ICON_output_atm'.

ICON's first work PE writes every namelist group it read, with all defaults and the readers'
own resets, to 'NAMELIST_ICON_output_atm' in the run directory, and closes the file before
ComIn's primary constructors run. The plugin reads it once ('load': rank 0 of ComIn's host
communicator reads, then broadcasts) and takes from it
- ICON's icon4py mode ('IconMode'), as ICON's icon4py dispatcher decides it: the mode is
  'icon4py_mode' if 'luse_icon4py_diffusion', else OFF; the plugin computes iff
  'icon4py_interface' selects ComIn and the mode is SUBSTITUTE or VERIFY;
- the configuration values of 'grid_init', the backend, and the two values that icon4py's
  'Diffusion' takes besides its 'DiffusionConfig' ('ARGUMENTS'), each from the namelist entry
  that icon4py's own configuration reads for it (the keys of
  'VerticalGridConfig.from_fortran_dict'; 'ndyn_substeps' of 'nonhydrostatic_nml'); ICON uses 5
  times 'nudge_max_coeff' (its reader scales it);
- for icon4py's 'DiffusionConfig', which reads its entries itself ('from_fortran_dict'), the
  entries that decide ICON's value of the field it does not read from ICON, 'loutshs'
  ('Loutshs'): no namelist entry; ICON's NWP physics set it at run time to whether the output
  lists 'ddt_tke_hsh', which the plugin takes from ICON's variables in the secondary
  constructor;
- ICON's condition for the extra diffusion call before the first dynamics step ('InitialCall'),
  against which the plugin checks the initial-call flag it derives from the entry points.

The reader ('parse') understands the list-directed namelist output of ICON's compilers:
'&GROUP' ... '/' (or '&END'), 'KEY = value,' with arrays continued on the following lines,
repeat counts 'r*value', logicals 'T'/'F' (also '.TRUE.'/'.FALSE.'), integers, reals in E, F
or D format (also with a three-digit exponent and no letter, '0.1-100'), and quoted strings. A
group may occur more than once (e.g. 'output_nml'). Names are lower-cased, as f90nml does.
The reader is lenient, so that the plugin does not depend on groups it does not read: a value
it cannot read (e.g. a complex number) is kept as 'Unreadable', and a group it cannot read
(e.g. an unclosed one) is skipped up to the next group and kept as a problem; both raise only
when the plugin asks for that value or group ('Namelists.problems' lists them).

Precision: reals are read with 'float', so a value written with 17 significant digits reads
back bit for bit. nvfortran writes 16 in F format (its form for magnitudes from 0.1), which is
exact for a value that is the nearest double to a decimal of at most 15 significant digits
(every namelist literal and every default written as one), but may be one unit in the last
place off for another value. The reader marks the reals written with 16 significant digits
whose value is no shorter decimal ('Namelists.maybe_inexact'); a value ICON computed that lies
within half a unit of the 16th digit of a shorter decimal cannot be told from the text.

Known limitation: ICON perturbs 'a_hshr' in ensemble runs after writing the file (and again
during the run with time-dependent perturbations); the namelist value is the unperturbed one.
"""

import dataclasses
import math
import os
import pathlib
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Final

from icon4py.model.common.utils import fortran_config


@dataclasses.dataclass(frozen=True)
class Unreadable:
    """A value the reader cannot read (kept, so that only a use of it fails)."""

    text: str


Value = bool | int | float | str | Unreadable | None
"""A namelist value; 'None' is a null value ('r*' without a value)."""

NAMELIST_FILE: Final = fortran_config.NAMELIST_ATM_FNAME
"""ICON's namelist output of the atmosphere ('NAMELIST_ICON_output_atm'), in the run directory."""

OFF, SUBSTITUTE, VERIFY = 0, 1, 2
MODE_NAMES: Final[Mapping[int, str]] = {OFF: "OFF", SUBSTITUTE: "SUBSTITUTE", VERIFY: "VERIFY"}
"""ICON's icon4py modes ('icon4py_mode' of 'run_nml')."""
INTERFACE_PY2FGEN, INTERFACE_COMIN = 0, 1
"""ICON's icon4py interfaces ('icon4py_interface' of 'run_nml')."""


# ---- the reader ---------------------------------------------------------------------------------

_TOKEN: Final = re.compile(
    r"""
      (?P<space>[\s,]+)
    | &(?P<group>[A-Za-z_]\w*)
    | (?P<key>[A-Za-z_][\w%]*(?:\([^)]*\))?)\s*=
    | (?P<end>/)
    | (?P<repeat>\d+)\*
    | '(?P<squote>(?:[^']|'')*)'
    | "(?P<dquote>(?:[^"]|"")*)"
    | (?P<bare>[^\s,/'"=&]+)
    """,
    re.VERBOSE,
)
_INT: Final = re.compile(r"[+-]?\d+")
_EXPONENT_WITHOUT_LETTER: Final = re.compile(r"([+-]?(?:\d+\.\d*|\.\d+|\d+))([+-]\d{3})")
"""A real with a three-digit exponent, which Fortran's E format writes without the letter."""
_DIGITS: Final = re.compile(r"[+-]?(\d*)\.?(\d*)")
_TRUE: Final = frozenset({"t", ".t.", ".true.", "true"})
_FALSE: Final = frozenset({"f", ".f.", ".false.", "false"})


class NamelistError(ValueError):
    pass


class _GroupInGroup(NamelistError):
    """A group starts inside another one: the reader resumes at the new group."""


def _bare_value(token: str) -> Value:
    """The value of an unquoted token; 'Unreadable' if it is none of the kinds above."""
    lower = token.lower()
    if lower in _TRUE:
        return True
    if lower in _FALSE:
        return False
    if _INT.fullmatch(token):
        return int(token)
    match = _EXPONENT_WITHOUT_LETTER.fullmatch(lower)
    if match:
        lower = f"{match.group(1)}e{match.group(2)}"
    try:
        return float(lower.replace("d", "e"))
    except ValueError:
        return Unreadable(token)


def _mantissa_digits(text: str) -> str:
    """The digits of a real's mantissa, without leading zeros ('0.0650' -> '650')."""
    mantissa = re.split(r"[eEdD]|(?<=[\d.])[+-]", text.strip(), maxsplit=1)[0]
    match = _DIGITS.match(mantissa)
    return ((match.group(1) + match.group(2)) if match else "").lstrip("0")


def significant_digits(text: str) -> int:
    """The number of significant digits written in a real's text (trailing zeros count)."""
    return len(_mantissa_digits(text))


def maybe_inexact(text: str, value: float) -> bool:
    """
    Whether a real read from 'text' may differ from the value that was written: fewer than 17
    significant digits, and the value is no decimal of at most 15 significant digits (whose
    nearest double the 16 digits give back exactly).
    """
    if significant_digits(text) >= 17 or not math.isfinite(value) or value == 0.0:
        return False
    return len(_mantissa_digits(repr(value)).rstrip("0")) >= 16


_GROUP_START: Final = re.compile(r"(?m)^[ \t]*&")
"""A '&' that starts a line: where the reader resumes after a group it cannot read."""
OUTSIDE: Final = ""
"""The key of 'Parsed.problems' for text outside any group."""


@dataclasses.dataclass(frozen=True)
class Parsed:
    """What 'parse' read: the groups, the groups it could not read, the maybe-inexact reals."""

    groups: dict[str, list[dict[str, tuple[Value, ...]]]]
    problems: dict[str, str]
    """Per group name (OUTSIDE: text between groups): why it could not be read (the first
    problem). Such a group is skipped up to the next one; its instances are not in 'groups'."""
    inexact: dict[str, list[frozenset[str]]]
    """Per group name and instance (as in 'groups'): the entries with a real written with 16
    significant digits whose value is no shorter decimal ('maybe_inexact')."""


class _Parser:
    """The state of 'parse': one method per token kind of '_TOKEN'."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.groups: dict[str, list[dict[str, tuple[Value, ...]]]] = {}
        self.problems: dict[str, str] = {}
        self.inexact: dict[str, list[frozenset[str]]] = {}
        self.group: list[tuple[str, list[Value]]] | None = None
        """The entries of the open group, in file order (an entry may be written twice)."""
        self.group_name = ""
        self.group_inexact: set[str] = set()
        self.repeat: int | None = None

    def fail(self, message: str, kind: type[NamelistError] = NamelistError) -> NamelistError:
        line = self.text.count("\n", 0, self.pos) + 1
        return kind(f"{NAMELIST_FILE}, line {line}: {message}")

    def run(self) -> Parsed:
        while self.pos < len(self.text):
            try:
                match = _TOKEN.match(self.text, self.pos)
                if match is None:
                    raise self.fail(f"cannot read {self.text[self.pos : self.pos + 20]!r}")
                getattr(self, f"on_{match.lastgroup}")(match)
                self.pos = match.end()
            except NamelistError as error:
                self.skip_group(str(error), here=isinstance(error, _GroupInGroup))
        if self.group is not None:
            self.skip_group(str(self.fail(f"'&{self.group_name}' is not closed with '/'")))
        return Parsed(self.groups, self.problems, self.inexact)

    def skip_group(self, problem: str, here: bool = False) -> None:
        """
        Keep the first problem of the open group (or outside), and resume at the next '&' that
        starts a line: after the current position, or at it if a group starts 'here'.
        """
        name = self.group_name if self.group is not None else OUTSIDE
        self.problems.setdefault(name, problem)
        start = self.pos if here else self.pos + 1
        line_start = self.text.rfind("\n", 0, self.pos) + 1
        resume = len(self.text)
        for match in _GROUP_START.finditer(self.text, line_start):
            if match.end() - 1 >= start:
                resume = match.end() - 1
                break
        self.pos = resume
        self.group, self.group_name, self.group_inexact, self.repeat = None, "", set(), None

    def add(self, value: Value) -> None:
        if not self.group:
            raise self.fail(f"value {value!r} outside a namelist entry")
        self.group[-1][1].extend([value] * (1 if self.repeat is None else self.repeat))
        self.repeat = None

    def flush_repeat(self) -> None:
        if self.repeat is not None:  # 'r*' before the next entry: r null values
            self.add(None)

    def on_space(self, match: re.Match[str]) -> None:
        if "," in match.group():
            self.flush_repeat()

    def on_group(self, match: re.Match[str]) -> None:
        if self.group is not None and match.group("group").lower() == "end":
            self.on_end(match)  # '&END' closes a group, as '/' does
            return
        if self.group is not None:
            raise self.fail(f"'&{match.group('group')}' inside '&{self.group_name}'", _GroupInGroup)
        self.group, self.group_name = [], match.group("group").lower()
        self.group_inexact = set()

    def on_end(self, match: re.Match[str]) -> None:
        if self.group is None:
            raise self.fail("'/' outside a namelist group")
        self.flush_repeat()
        # ICON lists some variables twice in a NAMELIST statement (e.g. 'nrestart_streams' of
        # 'io_nml'), and the compiler writes them twice: the same variable, the same values
        closed: dict[str, tuple[Value, ...]] = {}
        for name, values in self.group:
            if name in closed and closed[name] != tuple(values):
                raise self.fail(f"entry {name!r} twice in '&{self.group_name}', with other values")
            closed[name] = tuple(values)
        self.groups.setdefault(self.group_name, []).append(closed)
        self.inexact.setdefault(self.group_name, []).append(frozenset(self.group_inexact))
        self.group, self.group_name = None, ""

    def on_key(self, match: re.Match[str]) -> None:
        if self.group is None:
            raise self.fail(f"entry {match.group('key')!r} outside a namelist group")
        self.flush_repeat()
        self.group.append((match.group("key").lower(), []))

    def on_repeat(self, match: re.Match[str]) -> None:
        self.repeat = int(match.group("repeat"))

    def on_squote(self, match: re.Match[str]) -> None:
        self.add(match.group("squote").replace("''", "'"))

    def on_dquote(self, match: re.Match[str]) -> None:
        self.add(match.group("dquote").replace('""', '"'))

    def on_bare(self, match: re.Match[str]) -> None:
        token = match.group("bare")
        value = _bare_value(token)
        self.add(value)
        if type(value) is float and maybe_inexact(token, value) and self.group:
            self.group_inexact.add(self.group[-1][0])


def parse(text: str) -> Parsed:
    """
    All namelist groups of 'text' ('Parsed.groups': group name -> its instances, in file
    order, each a mapping of the entry names to their values, a tuple, one element for a
    scalar), and what it could not read.
    """
    return _Parser(text).run()


@dataclasses.dataclass(frozen=True)
class Namelists:
    """ICON's namelist output, parsed."""

    groups: Mapping[str, Sequence[Mapping[str, tuple[Value, ...]]]]
    unread: Mapping[str, str] = dataclasses.field(default_factory=dict)
    """The groups the reader could not read, and why ('Parsed.problems')."""
    inexact: Mapping[str, Sequence[frozenset[str]]] = dataclasses.field(default_factory=dict)
    """The entries with a real that may differ from ICON's value ('Parsed.inexact')."""

    @classmethod
    def from_text(cls, text: str) -> "Namelists":
        parsed = parse(text)
        return cls(parsed.groups, parsed.problems, parsed.inexact)

    def group(self, name: str) -> Mapping[str, tuple[Value, ...]] | None:
        """The only instance of a group, or 'None' if ICON did not write it; raises if the
        reader could not read it."""
        if name.lower() in self.unread:
            raise NamelistError(
                f"'&{name}' could not be read ({self.unread[name.lower()]}), and the plugin"
                " needs it."
            )
        instances = self.groups.get(name.lower(), ())
        if len(instances) > 1:
            raise NamelistError(f"{NAMELIST_FILE}: '&{name}' occurs {len(instances)} times.")
        return instances[0] if instances else None

    def get(self, group: str, key: str) -> tuple[Value, ...] | None:
        """The values of one entry, or 'None' if the group or the entry is missing; raises if
        the reader could not read the group or a value of the entry."""
        entries = self.group(group)
        values = None if entries is None else entries.get(key.lower())
        unreadable = [v.text for v in values or () if isinstance(v, Unreadable)]
        if unreadable:
            raise NamelistError(
                f"{group}: {key} = {', '.join(unreadable)}: cannot read the value, and the"
                " plugin needs it."
            )
        return values

    def maybe_inexact(self, group: str, key: str) -> bool:
        """Whether a real of the entry (of the only instance of the group) may differ from
        ICON's value in the last bit ('maybe_inexact')."""
        instances = self.inexact.get(group.lower(), ())
        return len(instances) == 1 and key.lower() in instances[0]

    def problems(self) -> list[str]:
        """What the reader could not read: whole groups, and single values (with their entry)."""
        found = [
            f"&{name or '(outside a group)'}: {problem}" for name, problem in self.unread.items()
        ]
        for name, instances in self.groups.items():
            for entries in instances:
                for key, values in entries.items():
                    texts = [v.text for v in values if isinstance(v, Unreadable)]
                    if texts:
                        found.append(f"&{name}: {key} = {', '.join(texts)}")
        return found

    def fortran_dict(self) -> dict[str, Any]:
        """The layout of f90nml's 'Namelist.todict()': scalars as values, arrays as lists, a
        repeated group as a list of groups (what icon4py's 'from_fortran_dict' readers take)."""

        def entries(group: Mapping[str, tuple[Value, ...]]) -> dict[str, Any]:
            return {k: v[0] if len(v) == 1 else list(v) for k, v in group.items()}

        return {
            name: entries(instances[0]) if len(instances) == 1 else [entries(g) for g in instances]
            for name, instances in self.groups.items()
        }


def load(comm: Any, path: str | os.PathLike[str]) -> Namelists:
    """
    Rank 0 of 'comm' (an mpi4py communicator; 'None' for one process) reads 'path' and
    broadcasts the text; every rank parses it. A failure to read raises on every rank.
    """
    text: str | None = None
    error: str | None = None
    if comm is None or comm.Get_rank() == 0:
        try:
            text = pathlib.Path(path).read_text()
        except OSError as exc:
            error = f"cannot read ICON's namelist output {str(path)!r}: {exc}"
    if comm is not None:
        text, error = comm.bcast((text, error), root=0)
    if error is not None or text is None:
        raise RuntimeError(error or f"no text read from {str(path)!r}")
    return Namelists.from_text(text)


def host_comm(comin: Any) -> Any:
    """ComIn's host communicator (ICON's work PEs) as an mpi4py communicator."""
    from mpi4py import MPI  # noqa: PLC0415 [import-outside-top-level]: only inside ICON

    return MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())


# ---- ICON's mode ------------------------------------------------------------------------------


def _format(value: Value) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    return "missing" if value is None else str(value)


@dataclasses.dataclass(frozen=True)
class IconMode:
    """ICON's icon4py switches ('run_nml'); 'None' where ICON did not write the entry."""

    icon4py_interface: int | None
    luse_icon4py_diffusion: bool | None
    icon4py_mode: int | None

    @property
    def switches(self) -> str:
        return (
            f"icon4py_interface={_format(self.icon4py_interface)},"
            f" luse_icon4py_diffusion={_format(self.luse_icon4py_diffusion)},"
            f" icon4py_mode={_format(self.icon4py_mode)}"
        )

    @property
    def mode(self) -> int:
        """ICON's mode for the diffusion, as its dispatcher decides it."""
        if self.icon4py_mode is None or not self.luse_icon4py_diffusion:
            return OFF
        return self.icon4py_mode

    @property
    def name(self) -> str:
        return MODE_NAMES.get(self.mode, f"unknown ({self.mode})")

    @property
    def computes(self) -> bool:
        """This plugin computes the diffusion: ICON delegates it to ComIn."""
        return self.icon4py_interface == INTERFACE_COMIN and self.mode in (SUBSTITUTE, VERIFY)

    def describe(self, entry_point: str, copied_at: str | None = None) -> str:
        """
        The startup line: what ICON does and what this plugin does at 'entry_point'; in VERIFY,
        where it copies ICON's input ('copied_at'), before ICON computes.
        """
        head = f"ICON mode {self.name} ({self.switches})"
        if None in (self.icon4py_interface, self.luse_icon4py_diffusion, self.icon4py_mode):
            return (
                f"{head}: ICON has no icon4py switches, so it computes the horizontal"
                " diffusion itself; this plugin stays idle"
            )
        if self.mode == OFF:
            return f"{head}: ICON computes the horizontal diffusion; this plugin stays idle"
        if not self.computes:
            return (
                f"{head}: icon4py computes the horizontal diffusion through py2fgen, not"
                " ComIn; this plugin stays idle"
            )
        if self.mode == SUBSTITUTE:
            return (
                f"{head}: this plugin computes the horizontal diffusion of domain 1 at"
                f" {entry_point}; ICON skips its own"
            )
        return (
            f"{head}: ICON computes the horizontal diffusion and continues with its own result;"
            f" this plugin computes it too, at {entry_point} on its own copies of ICON's input"
            f" from {copied_at}, and compares its results with ICON's there; ICON compares"
            " nothing"
        )


def _single(values: tuple[Value, ...] | None) -> Value:
    return None if values is None or len(values) != 1 else values[0]


def icon_mode(namelists: Namelists) -> IconMode:
    def entry(key: str, kind: type) -> Any:
        value = _single(namelists.get("run_nml", key))
        if value is not None and type(value) is not kind:
            raise NamelistError(f"run_nml: {key} = {value!r}, expected a {kind.__name__}.")
        return value

    return IconMode(
        icon4py_interface=entry("icon4py_interface", int),
        luse_icon4py_diffusion=entry("luse_icon4py_diffusion", bool),
        icon4py_mode=entry("icon4py_mode", int),
    )


# ---- the configuration arguments ---------------------------------------------------------------

NUDGE_MAX_COEFF_FACTOR: Final = 5.0
"""ICON uses 5 times the namelist's 'nudge_max_coeff' (its reader scales it; 'interpol_nml')."""


@dataclasses.dataclass(frozen=True)
class Setting:
    """One configuration argument: the namelist entry (group, name) and ICON's use of it."""

    group: str
    name: str
    index: int | None = None
    """The element of a per-domain array (0: domain 1), or 'None' for a scalar entry."""
    derive: Callable[[Value], Value] | None = None
    """ICON's effective value from the namelist value (default: the value itself)."""
    fallback: tuple[str, str] | None = None
    """Another entry (group, name) that gives ICON's value when this one is not written."""


def _nudge(value: Value) -> Value:
    if type(value) is not float:
        return value  # reported as a type error by 'arguments'
    return NUDGE_MAX_COEFF_FACTOR * value


ARGUMENTS: Final[Mapping[str, Mapping[str, Setting]]] = {
    "grid_init": {
        # the keys of icon4py's 'VerticalGridConfig.from_fortran_dict'
        "lowest_layer_thickness": Setting("sleve_nml", "min_lay_thckn"),
        "model_top_height": Setting("sleve_nml", "top_height"),
        "stretch_factor": Setting("sleve_nml", "stretch_fac"),
        "flat_height": Setting("sleve_nml", "flat_height"),
        "rayleigh_damping_height": Setting("nonhydrostatic_nml", "damp_height", index=0),
        "backend": Setting("run_nml", "icon4py_backend"),
    },
    "diffusion_init": {
        # what icon4py's 'Diffusion' takes besides its DiffusionConfig (py2fgen's names)
        "ndyn_substeps": Setting("nonhydrostatic_nml", "ndyn_substeps"),
        "nudge_max_coeff": Setting("interpol_nml", "nudge_max_coeff", derive=_nudge),
        "backend": Setting("run_nml", "icon4py_backend"),
    },
}
"""The configuration values of the granule's functions and where ICON's values come from."""

INWP: Final = 3
"""ICON's 'iforcing' of the NWP physics ('inwp' in 'mo_impl_constants')."""

LOUTSHS_FIELD: Final = "loutshs"
"""The field of icon4py's DiffusionConfig that 'from_fortran_dict' does not read from ICON."""

LOUTSHS_VARIABLE: Final = "ddt_tke_hsh"
"""ICON's variable whose presence on a domain gives ICON's 'loutshs' with NWP physics."""

LOUTSHS_ENTRIES: Final[Mapping[str, tuple[Setting, type]]] = {
    "iforcing": (Setting("run_nml", "iforcing"), int),
    "ldynamics": (Setting("run_nml", "ldynamics"), bool),
    "l_scm_mode": (Setting("grid_nml", "l_scm_mode"), bool),
}
"""The namelist entries that decide ICON's 'loutshs' ('Loutshs'), and their Python types."""


@dataclasses.dataclass(frozen=True)
class Loutshs:
    """
    The namelist entries that decide ICON's 'loutshs' (its 'turbdiff_config'), and its
    configured value.

    'loutshs' is no namelist entry. ICON's configuration holds its default, true, which ICON
    resets to false in single-column runs without dynamics ('l_scm_mode' and not 'ldynamics';
    'mo_nml_crosscheck'): the configured value. With NWP physics ICON changes it at run time
    ('runtime_loutshs').
    """

    iforcing: int
    ldynamics: bool
    l_scm_mode: bool | None
    """Read only without dynamics; 'None' with dynamics, where it does not matter."""

    @property
    def configured(self) -> bool:
        """ICON's value after its namelists, before its physics initialisation."""
        return not (self.l_scm_mode and not self.ldynamics)


def runtime_loutshs(
    iforcing: int, configured: bool, exposed: Collection[tuple[str, int]], domain: int
) -> tuple[bool, str]:
    """
    ICON's 'loutshs' when it initialises icon4py's diffusion of 'domain', and why: 'configured'
    ('Loutshs.configured') without NWP physics; with NWP physics ('iforcing' = INWP), ICON's
    physics initialisation ('init_nwp_phy', before ICON initialises icon4py's diffusion) sets it
    to whether the output of the domain lists 'ddt_tke_hsh' ('var_in_output', after ICON's
    output dictionary). ICON creates its variable 'ddt_tke_hsh' of a domain under exactly that
    condition ('mo_nwp_phy_state'), so its presence among ICON's variables 'exposed' (name,
    domain; known from ComIn's secondary constructor on) gives ICON's value.
    """
    if iforcing != INWP:
        return configured, (
            f"{LOUTSHS_FIELD} {_format(configured)}: ICON's configured value (iforcing {iforcing},"
            " no NWP physics)"
        )
    value = (LOUTSHS_VARIABLE, domain) in exposed
    listed, has = ("lists", "has the") if value else ("does not list", "has no")
    return value, (
        f"{LOUTSHS_FIELD} {_format(value)}: with NWP physics ICON sets it to whether the output of"
        f" domain {domain} lists {LOUTSHS_VARIABLE}; it {listed} it (ICON {has} variable"
        f" {LOUTSHS_VARIABLE} on domain {domain}); configured {_format(configured)}"
    )


def loutshs(namelists: Namelists) -> tuple[Loutshs | None, list[str]]:
    """The entries that decide ICON's 'loutshs'; 'None' and the problems if unreadable."""
    values: dict[str, Value] = {}
    errors: list[str] = []
    for name, (setting, kind) in LOUTSHS_ENTRIES.items():
        if name == "l_scm_mode" and values.get("ldynamics", False):
            values[name] = None  # it matters only without dynamics
            continue
        try:
            values[name] = argument(namelists, setting, kind)
        except NamelistError as error:
            errors.append(f"DiffusionConfig.{LOUTSHS_FIELD}: '{name}': {error}")
    if errors:
        return None, errors
    return Loutshs(**values), []  # type: ignore[arg-type]  # the kinds are checked above


def argument(namelists: Namelists, setting: Setting, kind: type) -> Value:
    """ICON's value of one configuration argument, of Python type 'kind' (bool, int, float)."""
    values = namelists.get(setting.group, setting.name)
    where = f"{setting.group}: {setting.name}"
    if values is None and setting.fallback is not None:
        values = namelists.get(*setting.fallback)
        where = f"{setting.fallback[0]}: {setting.fallback[1]}"
    if values is None:
        raise NamelistError(f"{NAMELIST_FILE} has no {where}.")
    if setting.index is None:
        if len(values) != 1:
            raise NamelistError(f"{where}: {len(values)} values, expected one.")
        value = values[0]
    else:
        if len(values) <= setting.index:
            raise NamelistError(f"{where}: no element {setting.index + 1}.")
        value = values[setting.index]
    if setting.derive is not None:
        value = setting.derive(value)
    if type(value) is not kind:
        raise NamelistError(f"{where} = {value!r}, expected a {kind.__name__}.")
    return value


def source(namelists: Namelists, setting: Setting) -> tuple[str, str]:
    """The entry (group, name) that gives ICON's value of 'setting' ('argument' reads it)."""
    if setting.fallback is not None and namelists.get(setting.group, setting.name) is None:
        return setting.fallback
    return (setting.group, setting.name)


def inexact_arguments(namelists: Namelists, function: str, names: Sequence[str]) -> list[str]:
    """The real configuration arguments 'names' of 'function' whose value in the namelist
    output may differ from ICON's in the last bit ('Namelists.maybe_inexact'), as
    'argument (group: entry)'."""
    table = ARGUMENTS.get(function, {})
    found = []
    for name in names:
        if name in table:
            group, entry = source(namelists, table[name])
            if namelists.maybe_inexact(group, entry):
                found.append(f"{name} ({group}: {entry})")
    return found


def arguments(
    namelists: Namelists, function: str, kinds: Mapping[str, type]
) -> tuple[dict[str, Value], list[str]]:
    """
    ICON's values of the configuration arguments 'kinds' (name -> Python type) of 'function';
    returns the values and the problems (one per argument that cannot be read).
    """
    table = ARGUMENTS.get(function, {})
    values: dict[str, Value] = {}
    errors: list[str] = []
    for name, kind in kinds.items():
        if name not in table:
            errors.append(f"{function}: '{name}' has no namelist entry.")
            continue
        try:
            values[name] = argument(namelists, table[name], kind)
        except NamelistError as error:
            errors.append(f"{function}: '{name}': {error}")
    return values, errors


# ---- the initial diffusion call ------------------------------------------------------------------

MODE_IAU: Final = 5
"""ICON's 'init_mode' of the incremental analysis update ('mo_impl_constants'); no initial call."""

INITIAL_CALL: Final[Mapping[str, tuple[Setting, type]]] = {
    "ldynamics": (Setting("run_nml", "ldynamics"), bool),
    "ltestcase": (Setting("run_nml", "ltestcase"), bool),
    "lhdiff_vn": (Setting("diffusion_nml", "lhdiff_vn"), bool),
    "init_mode": (Setting("initicon_nml", "init_mode"), int),
}
"""The namelist entries of ICON's condition for the initial diffusion call."""


@dataclasses.dataclass(frozen=True)
class InitialCall:
    """
    ICON's condition for the extra diffusion call of a real-data run before the first dynamics
    step of a domain ('integrate_nh' in 'mo_nh_stepping'): 'ldynamics', not 'ltestcase',
    'linit_dyn' (true in the first step of the domain, false after a restart), 'lhdiff_vn' and
    'init_mode' not IAU. ICON makes it before the dynamics, so it is the first diffusion call of
    that step; the regular one follows the dynamics.
    """

    ldynamics: bool
    ltestcase: bool
    lhdiff_vn: bool
    init_mode: int
    lrestartrun: bool = False
    """From ComIn's descriptive data, not from the namelist output."""

    def expected(self, step: int, call_of_step: int) -> bool:
        """Whether the diffusion call 'call_of_step' (from 1) of 'step' (from 1) is the initial one."""
        return (
            self.ldynamics
            and not self.ltestcase
            and self.lhdiff_vn
            and self.init_mode != MODE_IAU
            and not self.lrestartrun
            and step == 1
            and call_of_step == 1
        )

    def describe(self) -> str:
        return (
            f"ldynamics {_format(self.ldynamics)}, ltestcase {_format(self.ltestcase)},"
            f" lhdiff_vn {_format(self.lhdiff_vn)}, init_mode {self.init_mode},"
            f" lrestartrun {_format(self.lrestartrun)}"
        )


def initial_call(namelists: Namelists) -> tuple[InitialCall | None, list[str]]:
    """ICON's condition for the initial diffusion call; 'None' and the problems if unreadable."""
    values: dict[str, Value] = {}
    errors: list[str] = []
    for name, (setting, kind) in INITIAL_CALL.items():
        try:
            values[name] = argument(namelists, setting, kind)
        except NamelistError as error:
            errors.append(f"initial call: '{name}': {error}")
    if errors:
        return None, errors
    return InitialCall(**values), []  # type: ignore[arg-type]  # the kinds are checked above
