"""Benchmark comparing pymetis API paths for graph partitioning.
Usage:
    python bench_pymetis_api.py [--array-lib numpy|cupy] [--sizes 10000 80000] [--nparts 4]
"""

import argparse
import time

import numpy as np
import pymetis

parser = argparse.ArgumentParser()
parser.add_argument(
    "--array-lib",
    choices=["numpy", "cupy"],
    default="numpy",
    help="Library used to create the input adjacency array",
)
parser.add_argument(
    "--sizes",
    nargs="+",
    type=int,
    default=[10_000, 80_000, 2_600_000],
    help="Grid sizes (number of cells) to benchmark",
)
parser.add_argument(
    "--nparts",
    type=int,
    default=4,
    help="Number of partitions for METIS",
)
args = parser.parse_args()

if args.array_lib == "cupy":
    import cupy as xp

    def to_numpy(arr):
        return xp.asnumpy(arr)
else:
    xp = np

    def to_numpy(arr):
        return arr


def make_c2e2c_like(n_cells: int, neighbors: int = 3):
    """Generate a random 3-regular undirected graph with no self-loops.
    Ring edges connect (i, i-1) and (i, i+1), a random perfect matching
    supplies the 3rd neighbor. Requires n_cells to be even.
    """
    assert n_cells % 2 == 0, f"n_cells must be even for a 3-regular graph, got {n_cells}"
    rng = np.random.default_rng(42)
    adj_np = np.empty((n_cells, neighbors), dtype=np.int32)
    for i in range(n_cells):
        adj_np[i, 0] = (i - 1) % n_cells
        adj_np[i, 1] = (i + 1) % n_cells
    indices = np.arange(n_cells)
    rng.shuffle(indices)
    for k in range(0, n_cells, 2):
        i, j = int(indices[k]), int(indices[k + 1])
        adj_np[i, 2] = j
        adj_np[j, 2] = i
    return xp.asarray(adj_np)


def time_call(func, repeats=2, runs=2):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        for _ in range(repeats):
            func()
        elapsed = time.perf_counter() - start
        times.append(elapsed / repeats)
    return min(times)


def time_single(func):
    start = time.perf_counter()
    func()
    return time.perf_counter() - start


ARRAY_LIB = args.array_lib
SIZES = args.sizes
NPARTS = args.nparts

rows = []

for n_cells in SIZES:
    adj = make_c2e2c_like(n_cells)

    t_to_numpy = time_call(lambda: to_numpy(adj), repeats=20, runs=3)

    if n_cells <= 80_000:
        t_noconv = time_call(lambda: pymetis.part_graph(nparts=NPARTS, adjacency=adj))

        adj_np = to_numpy(adj)
        t_ndarray = time_call(lambda: pymetis.part_graph(nparts=NPARTS, adjacency=adj_np))

        adj_list = [row.tolist() for row in adj_np]
        t_list = time_call(lambda: pymetis.part_graph(nparts=NPARTS, adjacency=adj_list))

        def _csr_path():
            adjacency_matrix = to_numpy(adj)
            xadj = (
                np.arange(adjacency_matrix.shape[0] + 1, dtype=np.int32) * adjacency_matrix.shape[1]
            )
            adjncy = adjacency_matrix.ravel()
            return pymetis.part_graph(nparts=NPARTS, xadj=xadj, adjncy=adjncy)

        t_csr = time_call(_csr_path)
    else:
        t_noconv = time_single(lambda: pymetis.part_graph(nparts=NPARTS, adjacency=adj))

        adj_np = to_numpy(adj)
        t_ndarray = time_single(lambda: pymetis.part_graph(nparts=NPARTS, adjacency=adj_np))

        adj_list = [row.tolist() for row in adj_np]
        t_list = time_single(lambda: pymetis.part_graph(nparts=NPARTS, adjacency=adj_list))

        xadj = np.arange(n_cells + 1, dtype=np.int32) * adj_np.shape[1]
        adjncy = adj_np.ravel()
        t_csr = time_single(lambda: pymetis.part_graph(nparts=NPARTS, xadj=xadj, adjncy=adjncy))

    t_prep_list = time_call(lambda: [row.tolist() for row in to_numpy(adj)], repeats=20, runs=3)
    t_prep_csr = time_call(
        lambda: (
            np.arange(adj.shape[0] + 1, dtype=np.int32) * adj.shape[1],
            to_numpy(adj).ravel(),
        ),
        repeats=20,
        runs=3,
    )

    rows.append((n_cells, "adjacency=xp (no conv)", t_noconv, None))
    rows.append((n_cells, "adjacency=ndarray", t_ndarray, t_to_numpy))
    rows.append((n_cells, "adjacency=list", t_list, t_prep_list))
    rows.append((n_cells, "xadj/adjncy (CSR)", t_csr, t_prep_csr))

print(
    f"| {'array_lib':<10s} | {'n_cells':>8s} | {'nparts':>6s} | {'method':<24s} | {'total (ms)':>10s} | {'prep (ms)':>9s} |"
)
print(f"|{'-' * 12}|{'-' * 10}|{'-' * 8}|{'-' * 26}|{'-' * 12}|{'-' * 11}|")
for idx, (n_cells, method, t_total, t_prep) in enumerate(rows):
    prep_str = f"{t_prep * 1000:>9.4f}" if t_prep is not None else "        -"
    print(
        f"| {ARRAY_LIB:<10s} | {n_cells:>8d} | {NPARTS:>6d} | {method:<24s} | {t_total * 1000:>10.2f} | {prep_str} |"
    )
    if method == "xadj/adjncy (CSR)" and idx < len(rows) - 1:
        print(f"|{'-' * 12}|{'-' * 10}|{'-' * 8}|{'-' * 26}|{'-' * 12}|{'-' * 11}|")