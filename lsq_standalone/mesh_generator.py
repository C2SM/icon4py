# periodic
#
# 0v---0e-- 1v---3e-- 2v---6e-- 0v
# |  \ 0c   |  \ 1c   |  \2c
# |   \1e   |   \4e   |   \7e
# |2e   \   |5e   \   |8e   \
# |  3c   \ |   4c  \ |    5c\
# 3v---9e-- 4v--12e-- 5v--15e-- 3v
# |  \ 6c   |  \ 7c   |  \ 8c
# |   \10e  |   \13e  |   \16e
# |11e  \   |14e  \   |17e  \
# |  9c  \  |  10c \  |  11c \
# 6v--18e-- 7v--21e-- 8v--24e-- 6v
# |  \12c   |  \ 13c  |  \ 14c
# |   \19e  |   \22e  |   \25e
# |20e  \   |23e  \   |26e  \
# |  15c  \ | 16c   \ | 17c  \
# 0v       1v         2v        0v
#
import numpy

def mesh_generator():

    _VERTICES_SQRT = 128
    _VERTICES_SQRT = 32
    _VERTICES = _VERTICES_SQRT * _VERTICES_SQRT
    _CELLS = _VERTICES * 2
    _EDGES = _VERTICES * 3
    nodes = numpy.empty((_VERTICES_SQRT, _VERTICES_SQRT), dtype=int) 
    for j in range(_VERTICES_SQRT):
        for i in range(_VERTICES_SQRT):
            nodes[i, j] = i * _VERTICES_SQRT + j
    print("nodes =")
    print(nodes)
    c2v_table = numpy.empty((_CELLS, 3), dtype=int)
    for j in range(_VERTICES_SQRT):
        for i in range(_VERTICES_SQRT):
            c2v_table[i + j * _VERTICES_SQRT * 2, 0] = nodes[j, i]
            c2v_table[i + j * _VERTICES_SQRT * 2, 1] = nodes[j, (i + 1) % _VERTICES_SQRT]
            c2v_table[i + j * _VERTICES_SQRT * 2, 2] = nodes[(j + 1) % _VERTICES_SQRT, (i + 1) % _VERTICES_SQRT]
            c2v_table[i + j * _VERTICES_SQRT * 2 + _VERTICES_SQRT, 0] = nodes[j, i]
            c2v_table[i + j * _VERTICES_SQRT * 2 + _VERTICES_SQRT, 1] = nodes[(j + 1) % _VERTICES_SQRT, i]
            c2v_table[i + j * _VERTICES_SQRT * 2 + _VERTICES_SQRT, 2] = nodes[(j + 1) % _VERTICES_SQRT, (i + 1) % _VERTICES_SQRT]
    print("c2v_table =")
    print(c2v_table)
    e2v_table = numpy.empty((_EDGES, 2), dtype=int)
    for j in range(_VERTICES_SQRT):
        for i in range(_VERTICES_SQRT):
            e2v_table[(j * _VERTICES_SQRT + i) * 3 + 0, 0] = nodes[j, i]
            e2v_table[(j * _VERTICES_SQRT + i) * 3 + 0, 1] = nodes[j, (i + 1) % _VERTICES_SQRT]
            e2v_table[(j * _VERTICES_SQRT + i) * 3 + 1, 0] = nodes[j, i]
            e2v_table[(j * _VERTICES_SQRT + i) * 3 + 1, 1] = nodes[(j + 1) % _VERTICES_SQRT, (i + 1) % _VERTICES_SQRT]
            e2v_table[(j * _VERTICES_SQRT + i) * 3 + 2, 0] = nodes[j, i]
            e2v_table[(j * _VERTICES_SQRT + i) * 3 + 2, 1] = nodes[(j + 1) % _VERTICES_SQRT, i]
    print("e2v_table =")
    print(e2v_table)
    v2c_table = numpy.empty((_VERTICES, 6), dtype=int)
    for i in range(_VERTICES):
        k = 0
        for j in range(_CELLS):
            for l in range(3):
                if c2v_table[j, l] == i:
                    v2c_table[i, k] = j
                    k = k + 1
    print("v2c_table =")
    print(v2c_table)
    v2e_table = numpy.empty((_VERTICES, 6), dtype=int)
    for i in range(_VERTICES):
        k = 0
        for j in range(_EDGES):
            for l in range(2):
                if e2v_table[j, l] == i:
                    v2e_table[i, k] = j
                    k = k + 1
    print("v2e_table =")
    print(v2e_table)
    e2c_table = numpy.empty((_EDGES, 2), dtype=int)
    e2c_table[:, :] = -1
    for k in range(6):
        for j in range(6):
            for i in range(_EDGES):
                if v2c_table[e2v_table[i, 0], j] == v2c_table[e2v_table[i, 1], k]:
                    if e2c_table[i, 0] < 0:
                        e2c_table[i, 0] = v2c_table[e2v_table[i, 0], j]
                    else:
                        e2c_table[i, 1] = v2c_table[e2v_table[i, 0], j]
    print("e2c_table =")
    print(e2c_table)
    e2c2v_table = numpy.empty((_EDGES, 4), dtype=int)
    for i in range(_EDGES):
        for j in range(3):
            e2c2v_table[i, j] = c2v_table[e2c_table[i, 0], j]
        for j in range(3):
            l = 0
            for k in range(3):
                if e2c2v_table[i, k] != c2v_table[e2c_table[i, 1], j]:
                    l = l + 1
            if l == 3:
                e2c2v_table[i, 3] = c2v_table[e2c_table[i, 1], j]
    print("e2c2v_table =")
    print(e2c2v_table)
    c2e_table = numpy.empty((_CELLS, 3), dtype=int)
    c2e_table[:, :] = -1
    for k in range(6):
        for j in range(6):
            for i in range(_CELLS):
                if v2e_table[c2v_table[i, 0], j] == v2e_table[c2v_table[i, 1], k] or v2e_table[c2v_table[i, 0], j] == v2e_table[c2v_table[i, 2], k]:
                    if c2e_table[i, 0] < 0:
                        c2e_table[i, 0] = v2e_table[c2v_table[i, 0], j]
                    elif c2e_table[i, 1] < 0:
                        c2e_table[i, 1] = v2e_table[c2v_table[i, 0], j]
                    else:
                        c2e_table[i, 2] = v2e_table[c2v_table[i, 0], j]
                if v2e_table[c2v_table[i, 1], j] == v2e_table[c2v_table[i, 2], k]:
                    if c2e_table[i, 0] < 0:
                        c2e_table[i, 0] = v2e_table[c2v_table[i, 1], j]
                    elif c2e_table[i, 1] < 0:
                        c2e_table[i, 1] = v2e_table[c2v_table[i, 1], j]
                    else:
                        c2e_table[i, 2] = v2e_table[c2v_table[i, 1], j]
    print("c2e_table =")
    print(c2e_table)
    e2c2e0_table = numpy.empty((_EDGES, 5), dtype=int)
    for i in range(_EDGES):
        for j in range(3):
            e2c2e0_table[i, j] = c2e_table[e2c_table[i, 0], j]
        k = 3
        for j in range(3):
            if e2c2e0_table[i, 0] != c2e_table[e2c_table[i, 1], j] and e2c2e0_table[i, 1] != c2e_table[e2c_table[i, 1], j] and e2c2e0_table[i, 2] != c2e_table[e2c_table[i, 1], j]:
                e2c2e0_table[i, k] = c2e_table[e2c_table[i, 1], j]
                k = k + 1
    print("e2c2e0_table =")
    print(e2c2e0_table)
    e2c2e_table = numpy.empty((_EDGES, 4), dtype=int)
    for i in range(_EDGES):
        k = 0
        for j in range(5):
            if e2c2e0_table[i, j] != i:
                e2c2e_table[i, k] = e2c2e0_table[i, j]
                k = k + 1
    print("e2c2e_table =")
    print(e2c2e_table)
    c2e2cO_table = numpy.empty((_CELLS, 4), dtype=int)
    for i in range(_CELLS):
        for j in range(3):
            if e2c_table[c2e_table[i, j], 0] != i:
                c2e2cO_table[i, j] = e2c_table[c2e_table[i, j], 0]
            else:
                c2e2cO_table[i, j] = e2c_table[c2e_table[i, j], 1]
        c2e2cO_table[i, 3] = i
    print("c2e2cO_table =")
    print(c2e2cO_table)
    c2e2c_table = numpy.empty((_CELLS, 3), dtype=int)
    for i in range(_CELLS):
        for j in range(3):
            c2e2c_table[i, j] = c2e2cO_table[i, j]
    print("c2e2c_table =")
    print(c2e2c_table)
    c2e2c2e_table = numpy.empty((_CELLS, 9), dtype=int)
    for i in range(_CELLS):
        l = 0
        for j in range(3):
            for k in range(3):
                flag = True
                for m in range(l):
                    if c2e2c2e_table[i, m] == c2e_table[c2e2c_table[i, j], k]:
                        flag = False
                if flag:
                    c2e2c2e_table[i, l] = c2e_table[c2e2c_table[i, j], k]
                    l = l + 1
    print("c2e2c2e_table =")
    print(c2e2c2e_table)
    c2e2c2e2c_table = numpy.empty((_CELLS, 9), dtype=int)
    c2e2c2e2c_table[:, :] = -1
    for i in range(_CELLS):
        l = 0
        for j in range(9):
            for k in range(2):
                flag = True
                for m in range(l):
                    if c2e2c2e2c_table[i, m] == e2c_table[c2e2c2e_table[i, j], k]:
                        flag = False
                if flag and e2c_table[c2e2c2e_table[i, j], k] != i:
                    c2e2c2e2c_table[i, l] = e2c_table[c2e2c2e_table[i, j], k]
                    l = l + 1
    print("c2e2c2e2c_table =")
    print(c2e2c2e2c_table)
    
    cartesian_vertex_coordinates = numpy.empty((_VERTICES, 2), dtype=float)
    for i in range(_VERTICES):
        cartesian_vertex_coordinates[i, 0] = (i % _VERTICES_SQRT) - 0.5 * (i / _VERTICES_SQRT)
        cartesian_vertex_coordinates[i, 1] = - numpy.sqrt(3.0) * 0.5 * (i / _VERTICES_SQRT)
    print("cartesian_vertex_coordinates =")
    print(cartesian_vertex_coordinates)
    
    cartesian_cell_centers = numpy.empty((_CELLS, 2), dtype=float)
    cc = numpy.empty(3, dtype=float)
    for i in range(_CELLS):
        for j in range(3):
            if c2v_table[i, j] % _VERTICES_SQRT != 0 or c2v_table[i, (j + 1) % 3] % _VERTICES_SQRT == 1 or c2v_table[i, (j + 2) % 3] % _VERTICES_SQRT == 1:
                cc[j] = cartesian_vertex_coordinates[c2v_table[i, j], 0]
            else:
                cc[j] = cartesian_vertex_coordinates[c2v_table[i, j], 0] + _VERTICES_SQRT
        cartesian_cell_centers[i, 0] = (cc[0] + cc[1] + cc[2]) / 3
        for j in range(3):
            if c2v_table[i, j] / _VERTICES_SQRT != 0 or c2v_table[i, (j + 1) % 3] / _VERTICES_SQRT == 1 or c2v_table[i, (j + 2) % 3] / _VERTICES_SQRT == 1:
                cc[j] = cartesian_vertex_coordinates[c2v_table[i, j], 1]
            else:
                cc[j] = cartesian_vertex_coordinates[c2v_table[i, j], 1] - _VERTICES_SQRT * numpy.sqrt(3.0) * 0.5
        cartesian_cell_centers[i, 1] = (cc[0] + cc[1] + cc[2]) / 3
    print("cartesian_cell_centers =")
    print(cartesian_cell_centers)
    
    cartesian_edge_centers = numpy.empty((_EDGES, 2), dtype=float)
    ce = numpy.empty(2, dtype=float)
    for i in range(_EDGES):
        for j in range(2):
            if e2v_table[i, j] % _VERTICES_SQRT != 0 or e2v_table[i, (j + 1) % 2] % _VERTICES_SQRT == 1:
                ce[j] = cartesian_vertex_coordinates[e2v_table[i, j], 0]
            else:
                ce[j] = cartesian_vertex_coordinates[e2v_table[i, j], 0] + _VERTICES_SQRT
        cartesian_edge_centers[i, 0] = (ce[0] + ce[1]) / 2
        for j in range(2):
            if e2v_table[i, j] / _VERTICES_SQRT != 0 or e2v_table[i, (j + 1) % 2] / _VERTICES_SQRT == 1:
                ce[j] = cartesian_vertex_coordinates[e2v_table[i, j], 1]
            else:
                ce[j] = cartesian_vertex_coordinates[e2v_table[i, j], 1] - _VERTICES_SQRT * numpy.sqrt(3.0) * 0.5
        cartesian_edge_centers[i, 1] = (ce[0] + ce[1]) / 2
    print("cartesian_edge_centers =")
    print(cartesian_edge_centers)

    primal_edge_length = numpy.empty(_EDGES, dtype=float)
    primal_edge_length[:] = abs(cartesian_vertex_coordinates[e2v_table[i, j], 0] - cartesian_vertex_coordinates[e2v_table[i, j], 1])

    area = numpy.empty(_EDGES, dtype=float)
    # FIXME
    area[:] = numpy.square(primal_edge_length[:]) * numpy.sqrt(3.0) / 4.0

    edge_orientation = numpy.empty((_EDGES, 3), dtype=float)
    # FIXME
    edge_orientation[:, :] = 1

    return(nodes,
           c2v_table,
           e2v_table,
           v2c_table,
           v2e_table,
           e2c_table,
           e2c2v_table,
           c2e_table,
           e2c2e0_table,
           e2c2e_table,
           c2e2cO_table,
           c2e2c_table,
           c2e2c2e_table,
           c2e2c2e2c_table,
           cartesian_vertex_coordinates,
           cartesian_cell_centers,
           cartesian_edge_centers,
           primal_edge_length,
           edge_orientation,
           area
    )
