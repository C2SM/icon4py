# Naming convection of variables in ICON4Py

This naming convention should help to have a common style for variable names to improve the understandability of variables.

## Multiple-word identifier format

The variables must follow snake case.

## Structure of name

In a discretized domain, it is important to know where a variable is located.
ICON uses unstructured grid with the computational domain splitted into a finite number of triangular prisms. In the horizontal coordinates, all quantities in ICON sit either at cell centers, edges, or vertices. 
Currently, it is recommendded that we add a postfix, `_at_cell`, `_at_edge`, or `_at_vertex`, to a variable to let users understand the location at which the variable is defined. 

The vertical coordinate is divided into a discrete number of levels and a variable can either sit on the so-called full levels (model levels) or k-half levels (interface of two stacking neighboring triangular prisms). We add a prefix `khalf_` to show that a variable is defined on k-half levels. All other variables without the prefix of `khalf_` should be automatically understood to be defined on full levels.

The only exception is prognostic variables. We use the same name for prognostic variables as in the original ICON.
Feel free propose a better naming scheme.

## Example

The vertical wind speed on k-half levels with the contravariant correction due to the terrain-following coordinates is 

```
khalf_contravariant_corrected_w_at_cell
```
