from icon4py.common.dimension import CellDim, EdgeDim, VertexDim

 #*****************************************************************************
 #  We encode the grid as follows:
 #
 # \|{-1, 1}                 \|{0, 1}                  \|
 # -*-------------------------*-------------------------*-
 #  |\     {-1, 1, 0}         |\     {0, 1, 0}          |\
 #  | \                       | \                       |
 #  |  \                      |  \                      |
 #  |   \       {-1, 0, 1}    |   \       {0, 0, 1}     |
 #  |    \                    |    \                    |
 #  |     \                   |     \                   |
 #  |      \                  |      \                  |
 #  |       \                 |       \                 |
 #  |        \                |        \                |
 #  |         \               |         \               |
 #  |          \{-1, 0, 1}    |          \{0, 0, 1}     |
 #  |           \             |           \             |
 #  |            \            |            \            |
 #  |             \           |             \           |
 #  |{-1, 0, 2}    \          |{0, 0, 2}     \          |
 #  |               \         |               \         |
 #  |                \        |                \        |
 #  |                 \       |                 \       |
 #  |                  \      |                  \      |
 #  |                   \     |                   \     |
 #  |                    \    |                    \    |
 #  |   {-1, 0, 0}        \   |   {0, 0, 0}         \   |
 #  |                      \  |                      \  |
 #  |                       \ |                       \ |
 # \|{-1, 0}                 \|{0, 0}                  \|
 # -*-------------------------*-------------------------*-
 #  |\     {-1, 0, 0}         |\     {0, 0, 0}          |\
 #  | \                       | \                       |
 #  |  \                      |  \                      |
 #  |   \       {-1, -1, 1}   |   \       {0, -1, 1}    |
 #  |    \                    |    \                    |
 #  |     \                   |     \                   |
 #  |      \                  |      \                  |
 #  |       \                 |       \                 |
 #  |        \                |        \                |
 #  |         \               |         \               |
 #  |          \{-1, -1, 1}   |          \{0, -1, 1}    |
 #  |           \             |           \             |
 #  |            \            |            \            |
 #  |             \           |             \           |
 #  |{-1, -1, 2}   \          |{0, -1, 2}    \          |
 #  |               \         |               \         |
 #  |                \        |                \        |
 #  |                 \       |                 \       |
 #  |                  \      |                  \      |
 #  |                   \     |                   \     |
 #  |                    \    |                    \    |
 #  |   {-1, -1, 0}       \   |   {0, -1, 0}        \   |
 #  |                      \  |                      \  |
 #  |                       \ |                       \ |
 # \|{-1, -1}                \|{0, -1}                 \|
 # -*-------------------------*-------------------------*-
 #  |\     {-1, -1, 0}        |\     {0, -1, 0}         |\
 #
 #
 # Which is described by this general pattern:
 #
 #  |\
 #  | \
 #  |  \
 #  |   \       {x, y, 1}
 #  |    \
 #  |     \
 #  |      \
 #  |       \
 #  |        \
 #  |         \
 #  |          \{x, y, 1}
 #  |           \
 #  |            \
 #  |             \
 #  |{x, y, 2}     \
 #  |               \
 #  |                \
 #  |                 \
 #  |                  \
 #  |                   \
 #  |                    \
 #  |   {x, y, 0}         \
 #  |                      \
 #  |                       \
 #  |{x, y}                  \
 #  *-------------------------
 #         {x, y, 0}
 #
 # Note: Each location type uses a separate _id-space_.
 # {x, y, 0} can both mean an edge or cell. It's up to the user to ensure
 # they know what location type is meant.
 #/

class Connection:
    def __init__(self, start, end):
        self.start = start
        self.end = end

def vertex_to_edge(vertex):
    (x, y, _) = vertex
    return ((x, y, 0),
            (x, y, 2),
            (x - 1, y, 0),
            (x - 1, y, 1),
            (x, y - 1, 1),
            (x, y - 1, 2),
            )

def vertex_to_cell(vertex):
    (x, y, _) = vertex
    return ((x, y, 0),
            (x - 1, y, 0),
            (x - 1, y, 1),
            (x, y - 1, 0),
            (x, y - 1, 1),
            (x - 1, y - 1, 1),
            )

def edge_to_vertex(edge):
    (x, y, e) = edge
    if (e == 0):
        return ((x, y, 0), (x + 1, y, 0))
    elif (e == 1):
        return ((x + 1, y, 0), (x, y + 1, 0))
    elif (e == 2):
        return ((x, y, 0), (x, y + 1, 0))
    else:
        raise Exception("Invalid edge type")

def edge_to_cell(edge):
    (x, y, e) = edge
    if (e == 0):
        return ((x, y, 0), (x, y - 1, 1))
    elif (e == 1):
        return ((x, y, 0), (x, y, 1))
    elif (e == 2):
        return ((x, y, 0), (x - 1, y, 1))
    else:
        raise Exception("Invalid edge type")

def cell_to_vertex(cell):
    (x, y, c) = cell
    if (c == 0):
        return ((x, y, 0), (x + 1, y, 0), (x, y + 1, 0))
    elif (c == 1):
        return ((x + 1, y + 1, 0), (x + 1, y, 0), (x, y + 1, 0))
    else:
        raise Exception("Invalid edge type")

def cell_to_edge(cell):
    (x, y, c) = cell
    if (c == 0):
        return ((x, y, 0), (x, y, 1), (x, y, 2))
    elif (c == 1):
        return ((x, y, 1), (x + 1, y, 2), (x, y + 1, 0))
    else:
        raise Exception("Invalid edge type")

def ico_chain_size(chain):

    previous_location_type = chain[0]
    previous_locations = {(0, 0, 0)}

    for element in chain[1::]:
        current_location_type = element
        current_locations = set()
        assert(current_location_type != previous_location_type)
        connection = Connection(previous_location_type, current_location_type)

        if connection.start == VertexDim and connection.end == EdgeDim:
            for previous_location in previous_locations:
                neighbors = vertex_to_edge(previous_location)
                for neighbor in neighbors:
                    current_locations.add(neighbor)

        if connection.start == VertexDim and connection.end == CellDim:
            for previous_location in previous_locations:
                neighbors = vertex_to_cell(previous_location)
                for neighbor in neighbors:
                    current_locations.add(neighbor)

        if connection.start == EdgeDim and connection.end == VertexDim:
            for previous_location in previous_locations:
                neighbors = edge_to_vertex(previous_location)
                for neighbor in neighbors:
                    current_locations.add(neighbor)

        if connection.start == EdgeDim and connection.end == CellDim:
            for previous_location in previous_locations:
                neighbors = edge_to_cell(previous_location)
                for neighbor in neighbors:
                    current_locations.add(neighbor)

        if connection.start == CellDim and connection.end == VertexDim:
            for previous_location in previous_locations:
                neighbors = cell_to_vertex(previous_location)
                for neighbor in neighbors:
                    current_locations.add(neighbor)

        if connection.start == CellDim and connection.end == EdgeDim:
            for previous_location in previous_locations:
                neighbors = cell_to_edge(previous_location)
                for neighbor in neighbors:
                    current_locations.add(neighbor)

        previous_locations = current_locations
        previous_location_type = current_location_type

    return len(previous_locations)
