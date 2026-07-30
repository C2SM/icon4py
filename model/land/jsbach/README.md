# icon4py-land-jsbach

## Description

GT4Py port of the ICON-Land (JSBACH) land-surface scheme. The port targets the
`jsbach_lite` + TMX usecase and proceeds process by process; the first slice is
soil-snow energy (SSE). See `docs/sse_port_spec.md` for the SSE requirements
extracted from the Fortran source, and the design doc in the icon4py-knowledge
repo (`personal/jcanton/jsbach-port`) for the overall plan.

## Installation instructions

Check the `README.md` at the root of the `model` folder for installation instructions.
