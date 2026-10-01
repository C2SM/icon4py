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
- the configuration arguments of 'grid_init' and 'diffusion_init' ('ARGUMENTS'), each from the
  namelist entry that icon4py's own configuration reads for it (the 'IconOption's of
  'DiffusionConfig', the keys of 'VerticalGridConfig.from_fortran_dict'), with ICON's effective
  value where it differs from the namelist: ICON uses 5 times 'nudge_max_coeff' (its reader
  scales it), and 'loutshs' is no namelist entry (ICON's default is true, reset to false
  without dynamics);
- ICON's condition for the extra diffusion call before the first dynamics step ('InitialCall'),
  against which the plugin checks the initial-call flag it derives from the entry points.

The reader ('parse') understands the list-directed namelist output of ICON's compilers:
'&GROUP' ... '/', 'KEY = value,' with arrays continued on the following lines, repeat counts
'r*value', logicals 'T'/'F' (also '.TRUE.'/'.FALSE.'), integers, reals in E, F or D format
(read with 'float', so a value written with 17 significant digits reads back bit for bit;
nvfortran writes 16 in F format, which is exact for values from short namelist literals),
and quoted strings. A group may occur more than once (e.g. 'output_nml'). Names are
lower-cased, as f90nml does.

Known limitation: ICON perturbs 'a_hshr' in ensemble runs after writing the file (and again
during the run with time-dependent perturbations); the namelist value is the unperturbed one.
"""

import dataclasses
import os
import pathlib
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from icon4py.model.common.utils import fortran_config


Value = bool | int | float | str | None
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
_TRUE: Final = frozenset({"t", ".t.", ".true.", "true"})
_FALSE: Final = frozenset({"f", ".f.", ".false.", "false"})


class NamelistError(ValueError):
    pass


def _bare_value(token: str) -> Value:
    lower = token.lower()
    if lower in _TRUE:
        return True
    if lower in _FALSE:
        return False
    if _INT.fullmatch(token):
        return int(token)
    try:
        return float(lower.replace("d", "e"))
    except ValueError:
        raise NamelistError(f"cannot read the value {token!r}") from None


class _Parser:
    """The state of 'parse': one method per token kind of '_TOKEN'."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.groups: dict[str, list[dict[str, tuple[Value, ...]]]] = {}
        self.group: list[tuple[str, list[Value]]] | None = None
        """The entries of the open group, in file order (an entry may be written twice)."""
        self.group_name = ""
        self.repeat: int | None = None

    def fail(self, message: str) -> NamelistError:
        line = self.text.count("\n", 0, self.pos) + 1
        return NamelistError(f"{NAMELIST_FILE}, line {line}: {message}")

    def run(self) -> dict[str, list[dict[str, tuple[Value, ...]]]]:
        while self.pos < len(self.text):
            match = _TOKEN.match(self.text, self.pos)
            if match is None:
                raise self.fail(f"cannot read {self.text[self.pos : self.pos + 20]!r}")
            getattr(self, f"on_{match.lastgroup}")(match)
            self.pos = match.end()
        if self.group is not None:
            raise self.fail(f"'&{self.group_name}' is not closed with '/'")
        return self.groups

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
        if self.group is not None:
            raise self.fail(f"'&{match.group('group')}' inside '&{self.group_name}'")
        self.group, self.group_name = [], match.group("group").lower()

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
        self.group = None

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
        try:
            value = _bare_value(match.group("bare"))
        except NamelistError as error:
            raise self.fail(str(error)) from None
        self.add(value)


def parse(text: str) -> dict[str, list[dict[str, tuple[Value, ...]]]]:
    """
    All namelist groups of 'text': group name -> its instances (in file order), each a mapping
    of the entry names to their values (a tuple, one element for a scalar).
    """
    return _Parser(text).run()


@dataclasses.dataclass(frozen=True)
class Namelists:
    """ICON's namelist output, parsed."""

    groups: Mapping[str, Sequence[Mapping[str, tuple[Value, ...]]]]

    @classmethod
    def from_text(cls, text: str) -> "Namelists":
        return cls(parse(text))

    def group(self, name: str) -> Mapping[str, tuple[Value, ...]] | None:
        """The only instance of a group, or 'None' if ICON did not write it."""
        instances = self.groups.get(name.lower(), ())
        if len(instances) > 1:
            raise NamelistError(f"{NAMELIST_FILE}: '&{name}' occurs {len(instances)} times.")
        return instances[0] if instances else None

    def get(self, group: str, key: str) -> tuple[Value, ...] | None:
        """The values of one entry, or 'None' if the group or the entry is missing."""
        entries = self.group(group)
        return None if entries is None else entries.get(key.lower())

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

    @property
    def exposes_arguments(self) -> bool:
        """ICON's ComIn backend exposes the arguments (it checks 'icon4py_mode' alone)."""
        return self.icon4py_interface == INTERFACE_COMIN and self.icon4py_mode not in (None, OFF)

    def describe(self, entry_point: str) -> str:
        """The startup line: what ICON does and what this plugin does at 'entry_point'."""
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
            f" this plugin computes it too, at {entry_point} on ICON's copies of the input,"
            " and ICON compares the two"
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
    note: str = ""


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
        # the 'IconOption's of icon4py's 'DiffusionConfig'
        "ndyn_substeps": Setting("nonhydrostatic_nml", "ndyn_substeps"),
        "diffusion_type": Setting("diffusion_nml", "hdiff_order"),
        "hdiff_w": Setting("diffusion_nml", "lhdiff_w"),
        "hdiff_vn": Setting("diffusion_nml", "lhdiff_vn"),
        "hdiff_smag_w": Setting("diffusion_nml", "lhdiff_smag_w", index=0),
        "zdiffu_t": Setting("nonhydrostatic_nml", "l_zdiffu_t"),
        "type_t_diffu": Setting("diffusion_nml", "itype_t_diffu"),
        "type_vn_diffu": Setting("diffusion_nml", "itype_vn_diffu"),
        "hdiff_efdt_ratio": Setting("diffusion_nml", "hdiff_efdt_ratio"),
        "hdiff_w_efdt_ratio": Setting("diffusion_nml", "hdiff_w_efdt_ratio"),
        "smagorinski_scaling_factor": Setting("diffusion_nml", "hdiff_smag_fac"),
        "smagorinski_scaling_factor2": Setting("diffusion_nml", "hdiff_smag_fac2"),
        "smagorinski_scaling_factor3": Setting("diffusion_nml", "hdiff_smag_fac3"),
        "smagorinski_scaling_factor4": Setting("diffusion_nml", "hdiff_smag_fac4"),
        "smagorinski_scaling_height": Setting("diffusion_nml", "hdiff_smag_z"),
        "smagorinski_scaling_height2": Setting("diffusion_nml", "hdiff_smag_z2"),
        "smagorinski_scaling_height3": Setting("diffusion_nml", "hdiff_smag_z3"),
        "smagorinski_scaling_height4": Setting("diffusion_nml", "hdiff_smag_z4"),
        "hdiff_temp": Setting("diffusion_nml", "lhdiff_temp"),
        "denom_diffu_v": Setting("gridref_nml", "denom_diffu_v"),
        "nudge_max_coeff": Setting(
            "interpol_nml", "nudge_max_coeff", derive=_nudge, note="5 x the namelist value"
        ),
        "itype_sher": Setting("turbdiff_nml", "itype_sher"),
        "iforcing": Setting("run_nml", "iforcing"),
        "a_hshr": Setting("turbdiff_nml", "a_hshr"),
        # not a namelist entry: ICON's default is true, reset to false iff not 'ldynamics'
        "loutshs": Setting(
            "turbdiff_nml",
            "loutshs",
            fallback=("run_nml", "ldynamics"),
            note="ICON's default, false iff not ldynamics",
        ),
        "backend": Setting("run_nml", "icon4py_backend"),
    },
}
"""The configuration arguments of the exported functions and where ICON's values come from."""


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
