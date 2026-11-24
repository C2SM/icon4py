# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The inputs of 'grid_init' and 'diffusion_init' from ComIn's descriptive data of domain 1.

ICON fills ComIn's descriptive data (grid, decomposition, global data) before ComIn's primary
constructors; most arrays point into ICON's patch, some are copies. py2fgen gets the same
values in another layout, and the plugin maps one to the other ('ROUTES'):
- ICON passes the first block of its arrays ('(:,1)', '(:,1,:)'), which is all of them only if
  every patch has one block. ComIn's arrays keep the block axis ('[nproma, nblks(, n)]'), so the
  plugin takes the first block and asserts one block for cells, edges and vertices, one domain
  and domain 1 ('check').
- Connectivities are ICON's 1-based indices, as py2fgen passes them; 'e2c2e' is the edges'
  'quad_idx'; 'e2v' and 'e2c2v' are both the edges' four vertices (icon4py takes the first two
  for 'e2v').
- The owner masks of cells, edges and vertices are 'decomp_domain == 0' ('_owner_mask'): in
  ICON's decomposition of an atmosphere domain, halo level 0 holds exactly the cells, edges and
  vertices that the PE owns, and 'decomp_domain' is the halo level.
- 'comm_id' is ComIn's host communicator, ICON's work PEs ('comin.parallel_get_host_mpi_comm');
  py2fgen gets ICON's work communicator. The py2fgen probe compares the two with
  MPI_Comm_compare: IDENT or CONGRUENT, i.e. the same processes in the same order.
- The geometry: the edge normals at vertices and cells ('primal_normal_vert', 'dual_normal_vert',
  'primal_normal_cell', 'dual_normal_cell') are ComIn's copies '[nproma, nblks, n]' of ICON's
  vector components; py2fgen gets ICON's reordered copies '[nproma, n, nblks]'
  ('reorder_tangent_vectors_sparse'), of which the first block is the same '[nproma, n]'.
  Centres are ComIn's copies ('clat', 'clon', 'elat', 'elon'), 'vct_a' is the global copy.
- Derived: the inverse edge lengths ('inverse_primal_edge_lengths', 'inv_dual_edge_length') are
  '1 / length' where ICON computes the inverse and 0 elsewhere, as ICON stores them
  ('_inverse'); ComIn exposes the lengths, not the inverses. ICON initializes both to 0
  ('mo_alloc_patches.f90'), then computes them on its owned edges and copies them to the halo
  edges ('complete_patchinfo' in 'mo_intp_coeffs.f90'): the primal one on every edge, the dual
  one from the second row of the lateral boundary on, so that it stays 0 on the first row, i.e.
  on every edge whose 'refin_ctrl' is 1 (a halo edge has the 'refin_ctrl' of its owner). Both
  stay 0 on the padding.
- The interpolation coefficients of 'diffusion_init' point into ICON's interpolation state;
  their block axis is not the second one ('e_bln_c_s' '[nproma, 3, nblks]', 'geofac_grg'
  '[nproma, 4, nblks, 2]' with the x and y components last, the vertices' 'rbf_vec_coeff'
  '[6, 2, nproma, nblks]', which is py2fgen's 'rbf_vec_coeff_v').

The arrays are the plugin's own: a host copy, made once, and for each pass of the time loop
a fresh copy for the granule, on the device if py2fgen passes the argument there, else on the
host, viewed as a Field where py2fgen passes one, in py2fgen's layout (Fortran order). So
ICON's descriptive data is only read, never handed to the granule.
"""

import dataclasses
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, Final

import numpy as np

from icon4py.bindings.comin import _arguments, _views


DOMAIN_ID: Final = _arguments.DOMAIN_ID
BLOCK_AXIS: Final = 1
"""The block axis of ComIn's grid arrays '[nproma, nblks(, n)]'."""
KINDS: Final = ("cells", "edges", "verts")


@dataclasses.dataclass(frozen=True)
class Data:
    """ComIn's descriptive data: the global data, domain 1, and the ComIn module."""

    comin: Any
    glob: Any
    domain: Any


@dataclasses.dataclass(frozen=True)
class Route:
    """How an argument comes from the descriptive data."""

    source: str
    """The descriptive data it is taken from (for the log)."""
    value: Callable[[Data], Any]
    """The host value: an array in py2fgen's shape and dtype, or a scalar of py2fgen's type."""


def first_block(array: Any, block_axis: int) -> np.ndarray:
    """Drop ICON's block axis (index 0); EXCLAIM runs with one block (nproma >= local size)."""
    array = np.asarray(array)
    if array.shape[block_axis] != 1:
        raise ValueError(
            f"{array.shape[block_axis]} blocks along axis {block_axis} of shape {array.shape}:"
            " the granule's inputs are the first block, which is everything only for nblks == 1."
        )
    index: list[Any] = [slice(None)] * array.ndim
    index[block_axis] = 0
    return array[tuple(index)]


def _field(data: Data, kind: str, name: str) -> np.ndarray:
    return np.asarray(getattr(getattr(data.domain, kind), name))


def _first_block(
    kind: str, name: str, axis: int = BLOCK_AXIS, component: int | None = None
) -> Route:
    """The first block along 'axis'; then, if given, one 'component' of the last axis."""

    def value(data: Data) -> np.ndarray:
        block = first_block(_field(data, kind, name), axis)
        return np.array(block if component is None else block[..., component], order="F")

    where = "" if component is None else f", component {component + 1} of the last axis"
    return Route(f"{kind}.{name} (first block{where})", value)


def _whole(kind: str, name: str) -> Route:
    def value(data: Data) -> np.ndarray:
        return np.array(_field(data, kind, name), order="F")

    return Route(f"{kind}.{name}", value)


COUNT: Final[Mapping[str, str]] = {"cells": "ncells", "edges": "nedges", "verts": "nverts"}
"""The local number of cells, edges and vertices in the descriptive data."""


def _inverse(kind: str, name: str, skipped_row: int | None = None) -> Route:
    """
    1 / the first block on the real entries, 0 on the padding (ICON's initial value there) and,
    if 'skipped_row' is given, on the real entries whose 'refin_ctrl' is that row of the lateral
    boundary (ICON does not compute the inverse there either).
    """

    def value(data: Data) -> np.ndarray:
        length = first_block(_field(data, kind, name), BLOCK_AXIS)
        real = int(getattr(getattr(data.domain, kind), COUNT[kind]))
        inverse = np.zeros(length.shape, dtype=length.dtype, order="F")
        with np.errstate(divide="ignore"):  # a zero length gives inf, as in ICON
            inverse[:real] = 1.0 / length[:real]
        if skipped_row is not None:
            refin_ctrl = first_block(_field(data, kind, "refin_ctrl"), BLOCK_AXIS)
            inverse[:real][refin_ctrl[:real] == skipped_row] = 0.0
        return inverse

    zero = "" if skipped_row is None else f" and where {kind}.refin_ctrl == {skipped_row}"
    return Route(f"1 / {kind}.{name} (first block; 0 on padding{zero})", value)


def _global(name: str) -> Route:
    return Route(
        f"glob.{name}", lambda data: np.array(np.asarray(getattr(data.glob, name)), order="F")
    )


def _owner_mask(kind: str) -> Route:
    """ICON's owner mask of cells, edges or vertices: 'decomp_domain == 0'."""
    # ICON decomposes every atmosphere domain with halo order 1, which puts exactly the cells,
    # edges and vertices that the PE owns on halo level 0 (mo_setup_subdivision.f90:383-394,
    # 2324-2407; an MPI build requires at least one ghost row, mo_parallel_config.f90:273 in
    # the branch without NOMPI; a build without MPI has one PE, which owns everything), and
    # 'decomp_domain' is the halo level: the loops over the halo levels of the cells, edges and
    # vertices (mo_setup_subdivision.f90:1440-1447, 1540-1547, 1610-1617). ICON's owner_mask
    # is 'owner_local == p_pe_work' ('set_owner_mask', mo_model_domimp_patches.f90:824-838),
    # with 'owner_local' from the same halo-level lists (mo_setup_subdivision.f90:3500-3560).
    # So ICON's owner_mask equals 'decomp_domain == 0' for cells, edges and vertices.

    def value(data: Data) -> np.ndarray:
        return first_block(_field(data, kind, "decomp_domain"), BLOCK_AXIS) == 0

    return Route(f"{kind}.decomp_domain == 0 (first block)", value)


def _count(kind: str) -> Route:
    name = COUNT[kind]
    return Route(f"{kind}.{name}", lambda data: int(getattr(getattr(data.domain, kind), name)))


ROUTES: Final[Mapping[str, Mapping[str, Route]]] = {
    "grid_init": {
        **{
            f"{location}_{end}s": _whole(kind, f"{end}_index")
            for location, kind in (("cell", "cells"), ("vertex", "verts"), ("edge", "edges"))
            for end in ("start", "end")
        },
        "c2e": _first_block("cells", "edge_idx"),
        "e2c": _first_block("edges", "cell_idx"),
        "c2e2c": _first_block("cells", "neighbor_idx"),
        "e2c2e": _first_block("edges", "quad_idx"),
        "e2v": _first_block("edges", "vertex_idx"),
        "v2e": _first_block("verts", "edge_idx"),
        "v2c": _first_block("verts", "cell_idx"),
        "e2c2v": _first_block("edges", "vertex_idx"),
        "c2v": _first_block("cells", "vertex_idx"),
        "c_owner_mask": _owner_mask("cells"),
        "e_owner_mask": _owner_mask("edges"),
        "v_owner_mask": _owner_mask("verts"),
        "c_glb_index": _whole("cells", "glb_index"),
        "e_glb_index": _whole("edges", "glb_index"),
        "v_glb_index": _whole("verts", "glb_index"),
        "comm_id": Route(
            "parallel_get_host_mpi_comm()",
            lambda data: int(data.comin.parallel_get_host_mpi_comm()),
        ),
        "num_vertices": _count("verts"),
        "num_cells": _count("cells"),
        "num_edges": _count("edges"),
        "vertical_size": Route("domain.nlev", lambda data: int(data.domain.nlev)),
        "limited_area": Route("glob.l_limited_area", lambda data: bool(data.glob.l_limited_area)),
        # geometry
        "tangent_orientation": _first_block("edges", "tangent_orientation"),
        "inverse_primal_edge_lengths": _inverse("edges", "primal_edge_length"),
        "inv_dual_edge_length": _inverse("edges", "dual_edge_length", skipped_row=1),
        "inv_vert_vert_length": _first_block("edges", "inv_vert_vert_length"),
        "edge_areas": _first_block("edges", "area_edge"),
        "f_e": _first_block("edges", "f_e"),
        "cell_center_lat": _first_block("cells", "clat"),
        "cell_center_lon": _first_block("cells", "clon"),
        "cell_areas": _first_block("cells", "area"),
        **{
            f"{normal}_{x}": _first_block("edges", f"{normal}_{v}")
            for normal in (
                "primal_normal_vert",
                "dual_normal_vert",
                "primal_normal_cell",
                "dual_normal_cell",
            )
            for x, v in (("x", "v1"), ("y", "v2"))
        },
        "edge_center_lat": _first_block("edges", "elat"),
        "edge_center_lon": _first_block("edges", "elon"),
        "primal_normal_x": _first_block("edges", "primal_normal_v1"),
        "primal_normal_y": _first_block("edges", "primal_normal_v2"),
        "vct_a": _global("vct_a"),
        "mean_cell_area": Route("cells.mean_area", lambda data: float(data.domain.cells.mean_area)),
    },
    "diffusion_init": {
        "e_bln_c_s": _first_block("cells", "e_bln_c_s", axis=2),
        "geofac_div": _first_block("cells", "geofac_div", axis=2),
        "geofac_grg_x": _first_block("cells", "geofac_grg", axis=2, component=0),
        "geofac_grg_y": _first_block("cells", "geofac_grg", axis=2, component=1),
        "geofac_n2s": _first_block("cells", "geofac_n2s", axis=2),
        "nudgecoeff_e": _first_block("edges", "nudgecoeff_e"),
        "rbf_vec_coeff_v": _first_block("verts", "rbf_vec_coeff", axis=3),
    },
}
"""The inputs that ComIn's descriptive data provides, per function."""


def has_route(function: str, param: str) -> bool:
    return param in ROUTES.get(function, {})


class DescriptiveData:
    """
    The plugin's copies of the arguments from ComIn's descriptive data of domain 1: the host
    values, made once, and a fresh copy per call of 'argument' for the granule.
    """

    def __init__(self, comin: Any) -> None:
        self._comin = comin
        self._data: Data | None = None
        self._host: dict[tuple[str, str], Any] = {}

    @property
    def checked(self) -> bool:
        """Whether the descriptive data was read and checked."""
        return self._data is not None

    def data(self) -> Data:
        """The descriptive data, checked once ('check')."""
        if self._data is None:
            comin = self._comin
            data = Data(comin, comin.descrdata_get_global(), comin.descrdata_get_domain(DOMAIN_ID))
            check(data)
            self._data = data
        return self._data

    def describe(self) -> str:
        data = self.data()
        counts = ", ".join(
            f"{kind} {int(getattr(getattr(data.domain, kind), count))}"
            for kind, count in COUNT.items()
        )
        return (
            f"descriptive data: n_dom {int(data.glob.n_dom)}, domain {int(data.domain.id)},"
            f" nblks 1 for cells, edges and vertices; {counts}, nlev {int(data.domain.nlev)}"
        )

    def host(self, function: str, param: str) -> Any:
        """The host value of an argument (copied once)."""
        key = (function, param)
        if key not in self._host:
            self._host[key] = ROUTES[function][param].value(self.data())
        return self._host[key]

    def host_bytes(self) -> int:
        return sum(v.nbytes for v in self._host.values() if isinstance(v, np.ndarray))

    def argument(
        self, function: str, param: str, array: _arguments.ArrayParam | None, device_xp: Any
    ) -> Any:
        """
        The input as the granule gets it: a scalar as is; an array as a fresh copy of the host
        value, on the device if 'device_xp' is CuPy and the input is a device input, in Fortran
        order, as a Field if the input is a Field ('array' is the parameter; 'None' for a
        scalar).
        """
        value = self.host(function, param)
        if array is None:
            return value
        on_device = device_xp is not None and not array.host
        xp: ModuleType = device_xp if on_device else np
        copy = xp.array(value, order="F", copy=True)
        return copy if array.dims is None else _views.field_view(copy, array.dims)

    def release(self) -> None:
        self._host.clear()
        self._data = None


def check(data: Data) -> None:
    """One domain, domain 1, and one block of cells, edges and vertices (py2fgen's first block)."""
    problems = []
    if int(data.glob.n_dom) != 1:
        problems.append(f"n_dom is {int(data.glob.n_dom)}, expected 1 (no nesting)")
    if int(data.domain.id) != DOMAIN_ID:
        problems.append(f"the domain's id is {int(data.domain.id)}, expected {DOMAIN_ID}")
    for kind in KINDS:
        nblks = int(getattr(data.domain, kind).nblks)
        if nblks != 1:
            problems.append(f"{kind}.nblks is {nblks}, expected 1 (nproma >= the local count)")
    if problems:
        raise RuntimeError(
            "icon4py ComIn plugin: ComIn's descriptive data does not fit the arguments of"
            " icon4py's functions, which are the first block of domain 1: " + "; ".join(problems)
        )
