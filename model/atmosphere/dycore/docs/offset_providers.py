import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Final

SIDE: Final = 1  # Length of each triangle side
LABEL_TRIANGLES: Final = False  # Option to label each triangle center with its ID
COLORS: Final = list(mcolors.TABLEAU_COLORS.values()) #= ['lightblue', 'lightsalmon', 'lightseagreen', 'lightcoral']
AX_BORDER: Final = 0.05*SIDE  # Border around the axes

class Triangle:
    def __init__(self, ax, x0, y0, orientation):
        self.ax = ax
        self.x0 = x0
        self.y0 = y0
        self.orientation = orientation
        self.calculate_vertices()
        self.label_offset = 0.10 * SIDE
        self.cell_offset = 0.25 * SIDE
        self.edge_offset = 0.25 * SIDE
        self.vertex_size = 10
        self.bold_line = 6

    def calculate_vertices(self):
        if self.orientation == 'up':
            # Upward-pointing triangle
            self.A = (self.x0, self.y0)
            self.B = (self.x0 + SIDE, self.y0)
            self.C = (self.x0 + SIDE / 2, self.y0 + SIDE * np.sqrt(3) / 2)
        else:
            # Downward-pointing triangle
            self.A = (self.x0, self.y0)
            self.B = (self.x0 + SIDE / 2, self.y0 + SIDE * np.sqrt(3) / 2)
            self.C = (self.x0 - SIDE / 2, self.y0 + SIDE * np.sqrt(3) / 2)

        # Calculate the center
        self.CC = ((self.A[0] + self.B[0] + self.C[0]) / 3, (self.A[1] + self.B[1] + self.C[1]) / 3)

        # Calculate the midpoints of the edges
        self.AB = ((self.A[0] + self.B[0]) / 2, (self.A[1] + self.B[1]) / 2)
        self.BC = ((self.B[0] + self.C[0]) / 2, (self.B[1] + self.C[1]) / 2)
        self.CA = ((self.C[0] + self.A[0]) / 2, (self.C[1] + self.A[1]) / 2)

    def draw(self):
        # Draw the triangle edges
        triangle = plt.Polygon([self.A, self.B, self.C], edgecolor='black', fill=None)
        self.ax.add_patch(triangle)
    
    def print_labels(self, tri_id):
        """
        Labels the triangle with the given label.
        Parameters:
            label : label identifying the triangle
        """
        self.ax.text(self.CC[0], self.CC[1], f"T{tri_id}", fontsize=8, ha='center')
        # add labels to vertices depending on orientation
        if self.orientation == 'up':
            self.ax.text(self.A[0] + self.label_offset * np.cos(np.pi / 6), self.A[1] + self.label_offset * np.sin(np.pi / 6), 'A', fontsize=8, ha='center', va='center')
            self.ax.text(self.B[0] - self.label_offset * np.cos(np.pi / 6), self.B[1] + self.label_offset * np.sin(np.pi / 6), 'B', fontsize=8, ha='center', va='center')
            self.ax.text(self.C[0], self.C[1] - self.label_offset, 'C', fontsize=8, ha='center', va='center')
        else:
            self.ax.text(self.A[0], self.A[1] + self.label_offset, 'A', fontsize=8, ha='center', va='center')
            self.ax.text(self.B[0] - self.label_offset * np.cos(np.pi / 6), self.B[1] - self.label_offset * np.sin(np.pi / 6), 'B', fontsize=8, ha='center', va='center')
            self.ax.text(self.C[0] + self.label_offset * np.cos(np.pi / 6), self.C[1] - self.label_offset * np.sin(np.pi / 6), 'C', fontsize=8, ha='center', va='center')
    
    def color_vertex(self, vertex, coloridx=0):
        """
        Colors the vertex of the triangle.
        Parameters:
            vertex : Vertex to be colored ('A', 'B', or 'C')
        """
        vertex_size = self.vertex_size - 2*coloridx
        if vertex == 'A':
            self.ax.plot(self.A[0], self.A[1], 'o', color=COLORS[coloridx], markersize=vertex_size)
        elif vertex == 'B':
            self.ax.plot(self.B[0], self.B[1], 'o', color=COLORS[coloridx], markersize=vertex_size)
        elif vertex == 'C':
            self.ax.plot(self.C[0], self.C[1], 'o', color=COLORS[coloridx], markersize=vertex_size)

    def color_cell(self, coloridx=0):
        """
        Fills the triangle area with color, leaving some distance from the sides.
        """
        cell_offset = self.cell_offset + self.cell_offset/8*coloridx
        if self.orientation == 'up':
            A = (self.A[0] + cell_offset * np.cos(np.pi / 6), self.A[1] + cell_offset * np.sin(np.pi / 6))
            B = (self.B[0] - cell_offset * np.cos(np.pi / 6), self.B[1] + cell_offset * np.sin(np.pi / 6))
            C = (self.C[0], self.C[1] - cell_offset)
        else:
            A = (self.A[0], self.A[1] + cell_offset)
            B = (self.B[0] - cell_offset * np.cos(np.pi / 6), self.B[1] - cell_offset * np.sin(np.pi / 6))
            C = (self.C[0] + cell_offset * np.cos(np.pi / 6), self.C[1] - cell_offset * np.sin(np.pi / 6))

        # Draw the filled triangle
        filled_triangle = plt.Polygon([A, B, C], edgecolor=None, facecolor=COLORS[coloridx])
        self.ax.add_patch(filled_triangle)

    def color_edge(self, edge, coloridx=0):
        """
        Colors the specified edge of the triangle, making the line a bit bolder and leaving some distance from the vertices.
        Parameters:
            edge : Edge to be colored ('AB', 'BC', or 'CA')
        """
        edge_offset = self.edge_offset + self.edge_offset/4*coloridx
        if edge == 'AB':
            if self.orientation == 'up':
                V0 = (self.A[0] + edge_offset, self.A[1])
                V1 = (self.B[0] - edge_offset, self.B[1])
            else:
                V0 = (self.A[0] + edge_offset * np.cos(np.pi / 3), self.A[1] + edge_offset * np.sin(np.pi / 3))
                V1 = (self.B[0] - edge_offset * np.cos(np.pi / 3), self.B[1] - edge_offset * np.sin(np.pi / 3))
        elif edge == 'BC':
            if self.orientation == 'up':
                V0 = (self.B[0] - edge_offset * np.cos(np.pi / 3), self.B[1] + edge_offset * np.sin(np.pi / 3))
                V1 = (self.C[0] + edge_offset * np.cos(np.pi / 3), self.C[1] - edge_offset * np.sin(np.pi / 3))
            else:
                V0 = (self.B[0] - edge_offset, self.B[1])
                V1 = (self.C[0] + edge_offset, self.C[1])
        elif edge == 'CA':
            if self.orientation == 'up':
                V0 = (self.C[0] - edge_offset * np.cos(np.pi / 3), self.C[1] - edge_offset * np.sin(np.pi / 3))
                V1 = (self.A[0] + edge_offset * np.cos(np.pi / 3), self.A[1] + edge_offset * np.sin(np.pi / 3))
            else:
                V0 = (self.C[0] + edge_offset * np.cos(np.pi / 3), self.C[1] - edge_offset * np.sin(np.pi / 3))
                V1 = (self.A[0] - edge_offset * np.cos(np.pi / 3), self.A[1] + edge_offset * np.sin(np.pi / 3))
        
        self.ax.plot([V0[0], V1[0]], [V0[1], V1[1]], color=COLORS[coloridx], linewidth=self.bold_line - coloridx)

    def color_edges(self, coloridx=0):
        """
        Colors the edges of the triangle, making the lines a bit bolder and leaving some distance from the vertices.
        """
        self.color_edge('AB', coloridx)
        self.color_edge('BC', coloridx)
        self.color_edge('CA', coloridx)

#===============================================================================
def draw_arrow(ax, start, end, coloridx=0):
    """
    Draws an arrow from start to end coordinates.
    Parameters:
        ax      : Matplotlib axis object
        start   : Tuple of (x, y) for the start coordinates
        end     : Tuple of (x, y) for the end coordinates
        coloridx: Index for the color in COLORS
    """
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(facecolor=COLORS[coloridx], shrink=0.1, width=1, headwidth=10))

#===============================================================================
def draw_mesh(ax, nx, ny):
    """
    Draws a grid of triangles.
    Parameters:
        ax : Matplotlib axis object
    """
    triangles = []
    for y in range(ny):
        for x in range(nx):
            xA = x * SIDE - y*SIDE/2
            yA = y * SIDE * np.sqrt(3) / 2
            # Create upward-pointing triangle
            up_triangle = Triangle(ax, xA, yA, 'up')
            triangles.append(up_triangle)
            if x < nx - 1:
                # Create downward-pointing triangle
                down_triangle = Triangle(ax, xA + SIDE, yA, 'down')
                triangles.append(down_triangle)
        if y < ny - 1:
            # Create the two outer downward-pointing triangles
            xA = - y*SIDE/2
            down_triangle = Triangle(ax, xA, yA, 'down')
            triangles.append(down_triangle)
            xA += nx*SIDE
            down_triangle = Triangle(ax, xA, yA, 'down')
            triangles.append(down_triangle)
        nx += 1
    
    for i, T in enumerate(triangles):
        T.draw()
        if LABEL_TRIANGLES:
            T.print_labels(str(i))

    xlims = (-AX_BORDER -(ny-1) * SIDE/2, AX_BORDER + (nx-1)*SIDE - (ny-1)*SIDE/2)
    ylims = (-0.2, AX_BORDER + ny * SIDE * np.sqrt(3) / 2)

    return triangles, xlims, ylims

#===============================================================================
def add_legend(ax, label, xlims):
    """
    Adds a horizontal legend with rectangles colored with COLORS and labels from the input string.
    Parameters:
        ax    : Matplotlib axis object
        label : String containing the labels for the legend
        xlims : Tuple containing the x-axis limits for centering the legend
    """
    label = label.replace('2', '')  # Remove all '2' from the label
    label = label.replace('o', '')  # Remove possible 'o'
    N = len(label)
    rect_width = 0.2 * SIDE
    rect_height = 0.1 * SIDE
    spacing = 0.4 * SIDE

    # Calculate the starting x position to center the legend
    total_width = N * rect_width + (N - 1) * spacing
    legend_x = (xlims[1] - xlims[0] - total_width) / 2 + xlims[0]
    legend_y = -0.12  # Y position for the legend

    for i in range(N):
        # Draw the rectangle
        rect = plt.Rectangle((legend_x + i * (rect_width + spacing), legend_y), rect_width, rect_height, facecolor=COLORS[i])
        ax.add_patch(rect)

        # Print the label on top of the rectangle
        ax.text(legend_x + i * (rect_width + spacing) + rect_width / 2, legend_y + rect_height / 2, label[i], fontsize=12, ha='center', va='center', color='black')

        # Draw the arrow linking to the next rectangle
        if i < N - 1:
            start = (legend_x + i * (rect_width + spacing) + rect_width, legend_y + rect_height / 2)
            end = (legend_x + (i + 1) * (rect_width + spacing), legend_y + rect_height / 2)
            draw_arrow(ax, start, end, i)

#===============================================================================
def generate_mesh_figure(nx, ny, label):
    """
    Generates a figure with a grid of triangles.
    Parameters:
        nx : Number of triangles in the x direction (start raw)
        ny : Number of triangles in the y direction
    """
    fig = plt.figure(1); plt.clf(); plt.show(block=False)
    ax = fig.add_subplot(111)
    T, xlims, ylims = draw_mesh(ax, nx, ny)
    ax.set_title(f"Offset provider: {label}")
    add_legend(ax, label, xlims)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_aspect('equal')
    ax.axis('off')

    fname = f"_imgs/offsetProvider_{label}.png"
    fig.save = lambda: fig.savefig(fname, dpi=300, bbox_inches='tight')

    return fig, ax, T

#===============================================================================
# Plot
#

#-------------------------------------------------------------------------------
fig, ax, T = generate_mesh_figure(2, 2, "e2v")

Ta = T[1]
Ta.color_edge('BC')
draw_arrow(ax, Ta.BC, Ta.B)
draw_arrow(ax, Ta.BC, Ta.C)
Ta.color_vertex('B', 1)
Ta.color_vertex('C', 1)

fig.save()

#-------------------------------------------------------------------------------
fig, ax, T = generate_mesh_figure(2, 2, "e2c2e")

Ta = T[1]; Tb = T[7]
Ta.color_edge('BC')
draw_arrow(ax, Ta.BC, Ta.CC)
draw_arrow(ax, Tb.AB, Tb.CC)
Ta.color_cell(1)
Tb.color_cell(1)
draw_arrow(ax, Ta.CC, Ta.AB, 1)
draw_arrow(ax, Ta.CC, Ta.BC, 1)
draw_arrow(ax, Ta.CC, Ta.CA, 1)
draw_arrow(ax, Tb.CC, Tb.AB, 1)
draw_arrow(ax, Tb.CC, Tb.BC, 1)
draw_arrow(ax, Tb.CC, Tb.CA, 1)
Ta.color_edges(2)
Tb.color_edges(2)

fig.save()

#-------------------------------------------------------------------------------
fig, ax, T = generate_mesh_figure(2, 2, "c2e")

Ta = T[1]
Ta.color_cell()
draw_arrow(ax, Ta.CC, Ta.AB)
draw_arrow(ax, Ta.CC, Ta.BC)
draw_arrow(ax, Ta.CC, Ta.CA)
Ta.color_edges(1)

fig.save()

#-------------------------------------------------------------------------------
fig, ax, T = generate_mesh_figure(2, 2, "c2e2c")

Ta = T[1]; Tb = T[0]; Tc = T[2]; Td = T[7]
Ta.color_cell()
draw_arrow(ax, Ta.CC, Ta.AB)
draw_arrow(ax, Ta.CC, Ta.BC)
draw_arrow(ax, Ta.CC, Ta.CA)
Ta.color_edges(1)
draw_arrow(ax, Tb.BC, Tb.CC, 1)
draw_arrow(ax, Tc.CA, Tc.CC, 1)
draw_arrow(ax, Td.AB, Td.CC, 1)
Tb.color_cell(2)
Tc.color_cell(2)
Td.color_cell(2)

fig.save()

#-------------------------------------------------------------------------------
fig, ax, T = generate_mesh_figure(2, 2, "c2e2co")

Ta = T[1]; Tb = T[0]; Tc = T[2]; Td = T[7]
Ta.color_cell()
draw_arrow(ax, Ta.CC, Ta.AB)
draw_arrow(ax, Ta.CC, Ta.BC)
draw_arrow(ax, Ta.CC, Ta.CA)
Ta.color_edges(1)
draw_arrow(ax, Tb.BC, Tb.CC, 1)
draw_arrow(ax, Tc.CA, Tc.CC, 1)
draw_arrow(ax, Td.AB, Td.CC, 1)
Tb.color_cell(2)
Tc.color_cell(2)
Td.color_cell(2)
draw_arrow(ax, Ta.AB, Ta.CC, 1)
draw_arrow(ax, Ta.BC, Ta.CC, 1)
draw_arrow(ax, Ta.CA, Ta.CC, 1)
Ta.color_cell(2)

fig.save()

#===============================================================================
plt.show(block=False)