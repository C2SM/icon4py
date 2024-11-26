Description of symbols
======================

This page explains some of the symbols used in the numerical documentation.


$\sum_\offProv{c2e2c}$
----------------------
This symbol is used to indicate summation over a set of points.
The offset provider ($\offProv{c2e2c}$) is read left to right to get the indexes
of the source data but the data itself "moves" from right to left (as in the
figure).
In this example the source points are the three (green) cells adjacent to the
destination (blue) cell.
The (orange) edges are an intermediate step in the process.

.. image:: _imgs/offsetProvider_c2e2c.png
   :width: 350px
   :align: center
   :alt: c2e2c


$\Gradn_{\offProv{e2c}}$
------------------------
This symbol is used to indicate the difference (horizontal gradient) between two
points.
The offset provider ($\offProv{e2c}$) is read left to right to get the indexes
of the source data but the data itself "moves" from right to left (as in the
figure).
In this example source points are the two (orange) cells
adjacent to the destination (blue) edge.

.. image:: _imgs/offsetProvider_e2c.png
   :width: 350px
   :align: center
   :alt: e2c


$\nlev$
-------------
Number of (full) vertical levels.

$\nflatlev$
-------------
Number of flat vertical levels, located above the terrain-following levels.

$\nflatgradp$
-------------
Is the maximum height index at which the height of the center of an edge lies
within two neighboring cells.