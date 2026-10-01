# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of the arguments from ComIn's descriptive data ('_descrdata.py'): every route against the
argument ICON's py2fgen interface passes ('mo_icon4py_interfaces.f90', the first block of the
patch's arrays), on descriptive data laid out as ComIn's Python adapter hands it out (read-only
Fortran-order memoryviews with the block axis), plus the copies, the checks and the
communicator comparison.
"""

import functools
import subprocess
import sys
import types
from typing import Any

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.bindings import diffusion_wrapper, grid_wrapper
from icon4py.bindings.comin import _descrdata, _dual, _marshal
from icon4py.tools import py2fgen

from .test_comin_marshal import cupy_or_skip


NPROMA, NC, NE, NV, NLEV = 8, 5, 7, 4, 3
COUNTS = {"cells": NC, "edges": NE, "verts": NV}
RL = {"cells": 14, "edges": 24, "verts": 14}
"""The number of refinement-control levels of the index ranges (ICON's min_rl* .. max_rl*)."""
NEIGHBOURS = {
    ("cells", "edge_idx"): 3,
    ("cells", "neighbor_idx"): 3,
    ("cells", "vertex_idx"): 3,
    ("edges", "cell_idx"): 2,
    ("edges", "quad_idx"): 4,
    ("edges", "vertex_idx"): 4,
    ("verts", "edge_idx"): 6,
    ("verts", "cell_idx"): 6,
}
"""ComIn's connectivities '[nproma, nblks, n]' (ICON's 1-based indices)."""
FIELDS = {
    "cells": ("clat", "clon", "area"),
    "edges": (
        "tangent_orientation",
        "primal_edge_length",
        "dual_edge_length",
        "inv_vert_vert_length",
        "area_edge",
        "f_e",
        "elat",
        "elon",
        "primal_normal_v1",
        "primal_normal_v2",
    ),
}
"""ComIn's geometry fields '[nproma, nblks]' (copies or pointers into ICON's patch)."""
NORMALS = {
    "primal_normal_vert": 4,
    "dual_normal_vert": 4,
    "primal_normal_cell": 2,
    "dual_normal_cell": 2,
}
"""ComIn's copies '[nproma, nblks, n]' of the components v1, v2 of ICON's edge normals."""
INTERPOLATION = {
    ("cells", "e_bln_c_s"): (NPROMA, 3, 1),
    ("cells", "geofac_div"): (NPROMA, 3, 1),
    ("cells", "geofac_n2s"): (NPROMA, 4, 1),
    ("cells", "geofac_grg"): (NPROMA, 4, 1, 2),
    ("edges", "nudgecoeff_e"): (NPROMA, 1),
    ("verts", "rbf_vec_coeff"): (6, 2, NPROMA, 1),
}
"""ComIn's interpolation coefficients (pointers into ICON's interpolation state), their shapes."""
EDGE_REFIN_CTRL = np.array([1, 1, 2, 0, 3, 1, 0, 0], dtype=np.int32)
"""
The edges' 'refin_ctrl' (the row of the lateral boundary, 0 inside; a halo edge has the value of
its owner, here the last real one, 1) and the padding (0, ICON's initial value).
"""


def readonly(array: np.ndarray) -> memoryview:
    """As ComIn's Python adapter hands out an array: a read-only Fortran-order memoryview."""
    return memoryview(np.asfortranarray(array)).toreadonly()


class Icon:
    """ICON's patch of one PE in miniature (nblks == 1), as ComIn's descriptive data."""

    def __init__(self, seed: int = 1, nblks: int = 1, n_dom: int = 1, domain_id: int = 1):
        rng = np.random.default_rng(seed)
        self.arrays: dict[tuple[str, str], np.ndarray] = {}
        kinds = {}
        for kind, n in COUNTS.items():
            fields: dict[str, Any] = {
                {"cells": "ncells", "edges": "nedges", "verts": "nverts"}[kind]: n,
                "nblks": nblks,
            }
            self.add(fields, kind, "start_index", rng.integers(-5, NPROMA, RL[kind]))
            self.add(fields, kind, "end_index", rng.integers(-5, NPROMA, RL[kind]))
            self.add(fields, kind, "glb_index", rng.integers(1, 1000, n))
            decomp = np.full((NPROMA, 1), -1)
            decomp[:n, 0] = rng.integers(0, 3, n)
            decomp[0, 0] = 0
            self.add(fields, kind, "decomp_domain", decomp)
            kinds[kind] = fields
        for (kind, name), n in NEIGHBOURS.items():
            self.add(kinds[kind], kind, name, rng.integers(-1, NPROMA + 1, (NPROMA, 1, n)))
        for kind, names in FIELDS.items():
            for name in names:
                values = np.zeros((NPROMA, 1))  # ICON initializes the padding to 0
                values[: COUNTS[kind], 0] = rng.uniform(-2.0, 2.0, COUNTS[kind])
                self.add(kinds[kind], kind, name, values, np.float64)
        for normal, n in NORMALS.items():
            for v in ("v1", "v2"):
                self.add(
                    kinds["edges"],
                    "edges",
                    f"{normal}_{v}",
                    rng.normal(size=(NPROMA, 1, n)),
                    np.float64,
                )
        self.add(kinds["edges"], "edges", "refin_ctrl", EDGE_REFIN_CTRL.reshape(NPROMA, 1))
        self.icon_inverses()
        kinds["cells"]["mean_area"] = float(self.a("cells", "area")[:NC].mean())
        for (kind, name), shape in INTERPOLATION.items():
            self.add(kinds[kind], kind, name, rng.normal(size=shape), np.float64)
        ns = types.SimpleNamespace
        self.domain = ns(id=domain_id, nlev=NLEV, **{k: ns(**v) for k, v in kinds.items()})
        self.vct_a = np.linspace(22000.0, 0.0, NLEV + 1)
        self.glob = ns(n_dom=n_dom, l_limited_area=True, vct_a=readonly(self.vct_a))
        self.host_comm = 1140850688

    def icon_inverses(self) -> None:
        """
        ICON's inverse edge lengths ('complete_patchinfo'): 0 everywhere, then 1 / length on the
        edges from the first row of the lateral boundary on (primal) resp. from the second row on
        (dual), i.e. on every edge whose 'refin_ctrl' is not 1, owned or (by the halo exchange)
        not; the padding keeps 0.
        """
        refin_ctrl = self.a("edges", "refin_ctrl")[:NE, 0]
        for name, computed in (
            ("primal_edge_length", np.ones(NE, dtype=bool)),
            ("dual_edge_length", refin_ctrl != 1),
        ):
            inverse = np.zeros((NPROMA, 1))
            with np.errstate(divide="ignore"):
                inverse[:NE, 0][computed] = 1.0 / self.a("edges", name)[:NE, 0][computed]
            self.arrays["edges", f"inv_{name}"] = inverse

    def add(
        self, fields: dict[str, Any], kind: str, name: str, values: Any, dtype: Any = np.int32
    ) -> None:
        array = np.asfortranarray(np.asarray(values, dtype=dtype))
        self.arrays[kind, name] = array
        fields[name] = readonly(array)

    def comin(self) -> Any:
        return types.SimpleNamespace(
            descrdata_get_global=lambda: self.glob,
            descrdata_get_domain=lambda jg: self.domain,
            parallel_get_host_mpi_comm=lambda: self.host_comm,
        )

    def a(self, kind: str, name: str) -> np.ndarray:
        return self.arrays[kind, name]


def reorder_sparse(component: np.ndarray) -> np.ndarray:
    """ICON's 'reorder_tangent_vectors_sparse' (v(i, k, j) = field(i, j, k)), then py2fgen's first
    block '(:,:,1)'."""
    nproma, nblks, n = component.shape
    reordered = np.empty((nproma, n, nblks), order="F")
    for j in range(nblks):
        for k in range(n):
            reordered[:, k, j] = component[:, j, k]
    return reordered[:, :, 0]


PY2FGEN: dict[str, dict[str, Any]] = {
    # what ICON's py2fgen interface passes ('build_grid_init'; 'icon4py_fill_module_copies')
    "grid_init": {
        **{
            f"{loc}_{end}s": (lambda i, k=kind, e=end: i.a(k, f"{e}_index"))
            for loc, kind in (("cell", "cells"), ("vertex", "verts"), ("edge", "edges"))
            for end in ("start", "end")
        },
        "c2e": lambda i: i.a("cells", "edge_idx")[:, 0, :],
        "e2c": lambda i: i.a("edges", "cell_idx")[:, 0, :],
        "c2e2c": lambda i: i.a("cells", "neighbor_idx")[:, 0, :],
        "e2c2e": lambda i: i.a("edges", "quad_idx")[:, 0, :],
        "e2v": lambda i: i.a("edges", "vertex_idx")[:, 0, :],
        "v2e": lambda i: i.a("verts", "edge_idx")[:, 0, :],
        "v2c": lambda i: i.a("verts", "cell_idx")[:, 0, :],
        "e2c2v": lambda i: i.a("edges", "vertex_idx")[:, 0, :],
        "c2v": lambda i: i.a("cells", "vertex_idx")[:, 0, :],
        # ICON's owner masks, which ICON's decomposition makes decomp_domain == 0
        "c_owner_mask": lambda i: i.a("cells", "decomp_domain")[:, 0] == 0,
        "e_owner_mask": lambda i: i.a("edges", "decomp_domain")[:, 0] == 0,
        "v_owner_mask": lambda i: i.a("verts", "decomp_domain")[:, 0] == 0,
        "c_glb_index": lambda i: i.a("cells", "glb_index"),
        "e_glb_index": lambda i: i.a("edges", "glb_index"),
        "v_glb_index": lambda i: i.a("verts", "glb_index"),
        "comm_id": lambda i: i.host_comm,
        "num_vertices": lambda i: NV,
        "num_cells": lambda i: NC,
        "num_edges": lambda i: NE,
        "vertical_size": lambda i: NLEV,
        "limited_area": lambda i: True,
        "tangent_orientation": lambda i: i.a("edges", "tangent_orientation")[:, 0],
        "inverse_primal_edge_lengths": lambda i: i.a("edges", "inv_primal_edge_length")[:, 0],
        "inv_dual_edge_length": lambda i: i.a("edges", "inv_dual_edge_length")[:, 0],
        "inv_vert_vert_length": lambda i: i.a("edges", "inv_vert_vert_length")[:, 0],
        "edge_areas": lambda i: i.a("edges", "area_edge")[:, 0],
        "f_e": lambda i: i.a("edges", "f_e")[:, 0],
        "cell_center_lat": lambda i: i.a("cells", "clat")[:, 0],
        "cell_center_lon": lambda i: i.a("cells", "clon")[:, 0],
        "cell_areas": lambda i: i.a("cells", "area")[:, 0],
        **{
            f"{normal}_{x}": (lambda i, name=f"{normal}_{v}": reorder_sparse(i.a("edges", name)))
            for normal in NORMALS
            for x, v in (("x", "v1"), ("y", "v2"))
        },
        "edge_center_lat": lambda i: i.a("edges", "elat")[:, 0],
        "edge_center_lon": lambda i: i.a("edges", "elon")[:, 0],
        "primal_normal_x": lambda i: i.a("edges", "primal_normal_v1")[:, 0],
        "primal_normal_y": lambda i: i.a("edges", "primal_normal_v2")[:, 0],
        "vct_a": lambda i: i.vct_a,
        "mean_cell_area": lambda i: i.domain.cells.mean_area,
    },
    # 'build_diffusion_init': p_int_state(jg)%...(:,:,1), geofac_grg(:,:,1,1|2), (:,1), (:,:,:,1)
    "diffusion_init": {
        "e_bln_c_s": lambda i: i.a("cells", "e_bln_c_s")[:, :, 0],
        "geofac_div": lambda i: i.a("cells", "geofac_div")[:, :, 0],
        "geofac_grg_x": lambda i: i.a("cells", "geofac_grg")[:, :, 0, 0],
        "geofac_grg_y": lambda i: i.a("cells", "geofac_grg")[:, :, 0, 1],
        "geofac_n2s": lambda i: i.a("cells", "geofac_n2s")[:, :, 0],
        "nudgecoeff_e": lambda i: i.a("edges", "nudgecoeff_e")[:, 0],
        "rbf_vec_coeff_v": lambda i: i.a("verts", "rbf_vec_coeff")[:, :, :, 0],
    },
}
REAL = {"grid_init": grid_wrapper.grid_init, "diffusion_init": diffusion_wrapper.diffusion_init}


def test_every_route_is_an_argument_with_an_expected_value():
    for function, routes in _descrdata.ROUTES.items():
        assert set(routes) <= set(REAL[function].param_descriptors), function
        assert set(routes) == set(PY2FGEN[function]), function


@pytest.mark.parametrize(
    "function, param", [(f, p) for f, routes in _descrdata.ROUTES.items() for p in routes]
)
def test_route_equals_py2fgen_argument(function, param):
    icon = Icon()
    data = _descrdata.DescriptiveData(icon.comin())
    value = data.host(function, param)
    expected = PY2FGEN[function][param](icon)
    descriptor = REAL[function].param_descriptors[param]
    if isinstance(descriptor, py2fgen.ArrayParamDescriptor):
        expected = np.asarray(expected)
        dtype = {"BOOL": np.bool_, "INT32": np.int32, "FLOAT64": np.float64}[descriptor.dtype.name]
        assert value.dtype == expected.dtype == dtype
        assert value.shape == expected.shape and value.ndim == descriptor.rank
        bits = [np.ascontiguousarray(a).view(np.uint8) for a in (value, expected)]
        assert np.array_equal(*bits)
        assert value.flags.f_contiguous  # py2fgen's layout
        for array in icon.arrays.values():
            assert not np.shares_memory(value, array)  # the plugin's own copy
    else:
        assert type(value) is {"BOOL": bool, "INT32": int, "FLOAT64": float}[descriptor.dtype.name]
        assert value == expected
    assert data.host(function, param) is value  # made once


def test_descriptive_data_is_checked():
    for kwargs, message in (
        (dict(nblks=2), "cells.nblks is 2, expected 1"),
        (dict(n_dom=2), "n_dom is 2, expected 1"),
        (dict(domain_id=2), "the domain's id is 2, expected 1"),
    ):
        data = _descrdata.DescriptiveData(Icon(**kwargs).comin())
        with pytest.raises(RuntimeError, match=message):
            data.host("grid_init", "c2e")
        assert not data.checked
    data = _descrdata.DescriptiveData(Icon().comin())
    assert data.describe() == (
        "descriptive data: n_dom 1, domain 1, nblks 1 for cells, edges and vertices;"
        f" cells {NC}, edges {NE}, verts {NV}, nlev {NLEV}"
    )
    assert data.checked


def signature(name: str) -> dict[str, _marshal.ArrayParam]:
    return {a.name: a for a in _marshal.signature(name, REAL[name]).arrays}


def check_argument(data, xp, name, param):
    arrays = signature(name)
    first = data.argument(name, param, arrays[param], xp)
    second = data.argument(name, param, arrays[param], xp)
    host = data.host(name, param)
    for value in (first, second):
        array = value.ndarray if isinstance(value, gtx.Field) else value
        assert isinstance(value, gtx.Field) == (arrays[param].dims is not None)
        assert type(array).__module__.split(".")[0] == (
            "numpy" if xp is np or arrays[param].is_host else "cupy"
        )
        assert array.flags.f_contiguous and array.shape == host.shape
        got = array if isinstance(array, np.ndarray) else array.get()
        assert np.array_equal(got, host) and not np.shares_memory(got, host)
    a, b = (v.ndarray if isinstance(v, gtx.Field) else v for v in (first, second))
    if isinstance(a, np.ndarray):
        assert not np.shares_memory(a, b)
    else:
        assert a.data.ptr != b.data.ptr
    return first


@pytest.mark.parametrize("on_device", [False, True], ids=["host", "device"])
def test_arguments_are_fresh_copies(on_device):
    xp = cupy_or_skip() if on_device else np
    data = _descrdata.DescriptiveData(Icon().comin())
    c2e = check_argument(data, xp, "grid_init", "c2e")  # a Field (MAYBE_DEVICE)
    assert tuple(c2e.domain.dims) == signature("grid_init")["c2e"].dims
    check_argument(data, xp, "grid_init", "cell_starts")  # HOST: numpy, plain
    check_argument(data, xp, "grid_init", "c_owner_mask")  # HOST, bool
    normal = check_argument(data, xp, "grid_init", "primal_normal_vert_x")  # a 2-D Field
    assert tuple(normal.domain.dims) == signature("grid_init")["primal_normal_vert_x"].dims
    check_argument(data, xp, "grid_init", "vct_a")  # global data
    grg = check_argument(data, xp, "diffusion_init", "geofac_grg_y")  # a component of a 4-D array
    assert grg.ndarray.shape == (NPROMA, 4)
    rbf = check_argument(data, xp, "diffusion_init", "rbf_vec_coeff_v")  # plain (MAYBE_DEVICE)
    assert rbf.shape == (6, 2, NPROMA)
    assert data.argument("grid_init", "num_cells", None, xp) == NC
    data.release()
    assert not data.checked


def test_inverse_lengths_follow_icon():
    """
    1 / length where ICON computes it (inf for a zero length, as ICON's division), 0 elsewhere:
    the primal one on every real edge, the dual one where 'refin_ctrl' is not 1; 0 on the padding.
    """
    icon = Icon()
    icon.a("edges", "dual_edge_length")[2, 0] = 0.0  # refin_ctrl 2: computed
    icon.a("edges", "dual_edge_length")[0, 0] = 0.0  # refin_ctrl 1: not computed
    icon.a("edges", "primal_edge_length")[1, 0] = 0.0  # refin_ctrl 1: computed (primal)
    icon.icon_inverses()
    data = _descrdata.DescriptiveData(icon.comin())
    dual = data.host("grid_init", "inv_dual_edge_length")
    primal = data.host("grid_init", "inverse_primal_edge_lengths")
    first_row = EDGE_REFIN_CTRL[:NE] == 1
    assert list(np.flatnonzero(first_row)) == [0, 1, 5]
    assert dual[2] == np.inf and np.all(dual[:NE][first_row] == 0.0) and np.all(dual[NE:] == 0.0)
    assert primal[1] == np.inf and np.all(primal[NE:] == 0.0)
    length = icon.a("edges", "dual_edge_length")[:NE, 0]
    computed = ~first_row & (length != 0)
    assert np.array_equal(dual[:NE][computed], 1.0 / length[computed])
    for value, name in ((dual, "inv_dual_edge_length"), (primal, "inv_primal_edge_length")):
        expected = icon.a("edges", name)[:, 0]
        assert np.array_equal(np.signbit(value), np.signbit(expected))  # +0, as ICON's
        assert np.array_equal(value, expected)
    assert _descrdata.ROUTES["grid_init"]["inv_dual_edge_length"].source == (
        "1 / edges.dual_edge_length (first block; 0 on padding and where edges.refin_ctrl == 1)"
    )
    assert _descrdata.ROUTES["grid_init"]["inverse_primal_edge_lengths"].source == (
        "1 / edges.primal_edge_length (first block; 0 on padding)"
    )


def test_owner_masks():
    """
    The owner masks of cells, edges and vertices are 'decomp_domain == 0', from the descriptive
    data that ComIn has (no owner mask of its own); False on the padding ('decomp_domain' -1).
    """
    icon = Icon()
    data = _descrdata.DescriptiveData(icon.comin())
    routes = _descrdata.ROUTES["grid_init"]
    for kind in ("cells", "edges", "verts"):
        param = f"{kind[0]}_owner_mask"
        assert routes[param].source == f"{kind}.decomp_domain == 0 (first block)"
        assert not hasattr(getattr(icon.domain, kind), "owner_mask")
        mask = data.host("grid_init", param)
        assert mask.dtype == np.bool_ and mask.shape == (NPROMA,)
        assert np.array_equal(mask, icon.a(kind, "decomp_domain")[:, 0] == 0)
        assert mask[0] and not mask[COUNTS[kind] :].any()


def test_domain_value_for_the_probe():
    icon = Icon()
    value = _descrdata.domain_value("grid_init", "c2e")(icon.domain)
    assert np.array_equal(value, icon.a("cells", "edge_idx")[:, 0, :])


# ---- the communicator --------------------------------------------------------------------------


@functools.cache
def mpi_usable() -> bool:
    """Whether MPI initializes here (a singleton; tried in a subprocess, as a failure aborts)."""
    probe = "from mpi4py import MPI; MPI.COMM_WORLD.Dup().Free()"
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, timeout=120, check=False
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def mpi_or_skip() -> Any:
    """mpi4py's MPI, initialized (icon4py's decomposition module leaves that to the host)."""
    mpi = pytest.importorskip("mpi4py.MPI")
    if not mpi.Is_initialized():
        if not mpi_usable():
            pytest.skip("MPI does not initialize in this environment.")
        mpi.Init()
    return mpi


def test_communicators_with_mpi():
    mpi = mpi_or_skip()
    world = mpi.COMM_WORLD
    duplicate = world.Dup()
    try:
        new, old = _descrdata.communicators(duplicate.py2f(), world.py2f())
        assert (new.relation, new.members, old.members) == ("CONGRUENT", (0,), (0,))
        assert old.relation is None and new.handle == duplicate.py2f()
        assert _dual.compare(new, old).identical
        same, _ = _descrdata.communicators(world.py2f(), world.py2f())
        assert same.relation == "IDENT"
    finally:
        duplicate.Free()


class FakeComm:
    def __init__(self, handle: int, members: tuple[int, ...]):
        self.handle, self.members = handle, members

    def Compare(self, other):
        if self.members == other.members:
            return 0 if self.handle == other.handle else 1
        return 2 if sorted(self.members) == sorted(other.members) else 3

    def Get_group(self):
        return FakeGroup(self.members)


class FakeGroup:
    def __init__(self, members):
        self.members = members

    def Get_size(self):
        return len(self.members)

    def Translate_ranks(self, ranks, world):
        return [self.members[r] for r in ranks]

    def Free(self):
        pass


@pytest.mark.parametrize(
    "new, relation, identical",
    [((0, 1), "CONGRUENT", True), ((1, 0), "SIMILAR", False), ((0, 2), "UNEQUAL", False)],
)
def test_communicators_differ(monkeypatch, new, relation, identical):
    comms = {7: FakeComm(7, new), 8: FakeComm(8, (0, 1))}
    fake = types.SimpleNamespace(
        IDENT=0,
        CONGRUENT=1,
        SIMILAR=2,
        UNEQUAL=3,
        Comm=types.SimpleNamespace(f2py=comms.__getitem__),
        COMM_WORLD=FakeComm(0, (0, 1, 2, 3)),
    )
    fake.Is_initialized, fake.Is_finalized = (lambda: True), (lambda: False)
    monkeypatch.setitem(sys.modules, "mpi4py", types.SimpleNamespace(MPI=fake))
    a, b = _descrdata.communicators(7, 8)
    assert a.relation == relation and a.members == new and b.members == (0, 1)
    assert _dual.compare(a, b).identical == identical
