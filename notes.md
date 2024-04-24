# notes

## icon4py

run:
/exclaim/jcanton/repos/icon4py/model/driver/src/icon4py/model/driver/dycore_driver.py

first argument = path to serialized data
experiment_name = jabw

```[bash]

python ./model/driver/src/icon4py/model/driver/dycore_driver.py ser_data icon_pydycore --experiment_name=gauss3d
```

namelists:
edit icon_configuration.py

## working on

there may be a bug in the update for w in solve_nonhydro.py line 1448
self.stencil_solve_tridiagonal_matrix_for_w_forward_sweep
returns non-zero w when we expect it to be zero

## pressure gradient on slopes

look at this stencil compute_horizontal_gradient_of_extner_pressure_for_multiple_levels and its corrector to see how that extracts pressure gradient at the surface and uses it further up