#!/bin/bash

rm imgs/*
rm ibm_*.csv
python model/driver/src/icon4py/model/driver/icon4py_driver.py testdata/ser_icondata/mpitask1/gauss3d_torus/ser_data --icon4py_driver_backend=gtfn_cpu --experiment_type=gauss3d_torus --grid_root=2 --grid_level=0 --enable_output
