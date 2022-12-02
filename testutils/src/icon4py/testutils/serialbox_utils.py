# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
from typing import List

from functional.common import Dimension
from functional.iterator.embedded import np_as_located_field
from serialbox import Savepoint

from icon4py.common.dimension import KDim, EdgeDim, ECVDim, CellDim, VertexDim

try:
    import serialbox as ser
except ImportError:
    external_src = os.path.join(os.path.dirname(__file__), "../../_external_src/")
    os.chdir(external_src)
    os.system(
        "git clone --recursive https://github.com/GridTools/serialbox; CC=`which gcc` CXX=`which g++` pip install serialbox/src/serialbox-python"
    )
    import serialbox as ser




class IconDiffusionSavepoint:
    def __init__(self, sp: Savepoint, ser: ser.Serializer):
        self.savepoint = sp
        self.serializer = ser

    def print_meta_info(self):
        print(self.savepoint.metainfo)

    def physical_height_field(self):
        return self._get_field("vct_a", KDim)

    def _get_field(self, name, *dimensions):
        return np_as_located_field(dimensions)(self.serializer.read(name, self.savepoint))

    def get_metadata(self, *names):
        metadata = self.savepoint.metainfo.to_dict()
        return {n:metadata[n] for n in names if n in metadata}

    def tangent_orientation(self):
        return self._get_field("tangent_orientation", EdgeDim)

    def inverse_primal_edge_lengths(self):
        return self._get_field("inverse_primal_edge_lengths", EdgeDim)

    def inv_vert_vert_length(self):
        return self._get_field("inv_vert_vert_length", EdgeDim)

    def primal_normal_vert_x(self):
        return self._get_field("primal_normal_vert_x", ECVDim)

    def primal_normal_vert_y(self):
        return self._get_field("primal_normal_vert_y", ECVDim)

    def dual_normal_vert_y(self):
        return self._get_field("dual_normal_vert_y", ECVDim)

    def dual_normal_vert_x(self):
        return self._get_field("dual_normal_vert_x", ECVDim)

    def cell_areas(self):
        return self._get_field("cell_areas", CellDim)

    def edge_areas(self):
        return self._get_field("cell_areas", EdgeDim)

    def inv_dual_edge_length(self):
        return self._get_field("inv_dual_edge_length", EdgeDim)

    def cells_start_index(self):
        return self.serializer.read("cells_start_index", self.savepoint)

    def cells_end_index(self):
        return self.serializer.read("c_end_index", self.savepoint)

    def vertex_start_index(self):
        return self.serializer.read("v_start_index", self.savepoint)

    def vertex_end_index(self):
        return self.serializer.read("v_end_index", self.savepoint)

    def edge_start_index(self):
        return self.serializer.read("e_start_index", self.savepoint)

    def edge_end_index(self):
        return self.serializer.read("e_end_index", self.savepoint)

    def refin_ctrl(self, dim: Dimension):
        if dim == CellDim:
            return self.serializer.read("c_refin_ctl", self.savepoint)
        elif dim == EdgeDim:
            return self.serializer.read("e_refin_ctl", self.savepoint)
        elif dim == VertexDim:
            return self.serializer.read("v_refin_ctl", self.savepoint)
        else:
            return None
    def c2e(self):
        return self.serializer.read("c2e", self.savepoint)

    def c2e2c(self):
        return self.serializer.read("c2e2c", self.savepoint)

    def e2c(self):
        return self.serializer.read("e2c", self.savepoint)
    def e2v(self):
        return self.serializer.read("e2v", self.savepoint)
class IconSerialDataProvider:

    def __init__(self, fname_prefix, path="."):
        self.rank = 0
        self.serializer: ser.Serializer = None
        self.file_path: str = path
        self.fname = f"{fname_prefix}_rank{str(self.rank)}"
        self._init_serializer()

    def _init_serializer(self):
        if not self.fname:
            print(" WARNING: no filename! closing serializer")
        self.serializer = ser.Serializer(ser.OpenModeKind.Read, self.file_path, self.fname)

    def print_info(self):
        print(f"SAVEPOINTS: {self.serializer.savepoint_list()}")
        print(f"FIELDNAMES: {self.serializer.fieldnames()}")

    def from_savepoint(self, linit:bool, date:str) -> IconDiffusionSavepoint:
        savepoint = (
            self.serializer.savepoint["call-diffusion-init"]
            .linit[linit]
            .date[date]
            .as_savepoint()
        )
        return IconDiffusionSavepoint(savepoint, self.serializer)



    def get_fields(self, metadata: List[str], fields: List[str]):
        savepoint = (
            self.serializer.savepoint["call-diffusion-init"]
            .linit[False]
            .date["2021-06-20T12:00:10.000"]
            .as_savepoint()
        )
        print(savepoint.metainfo)
        meta_present = {}
        meta_absent = []
        for md in metadata:
            if md in savepoint.metainfo.to_dict():
                meta_present[md] = savepoint.metainfo[md]
            else:
                meta_absent.append(md)

        fields_present = {}
        fields_absent = []
        for field_name in fields:
            if field_name in self.serializer.fieldnames():
                fields_present[field_name] = self.serializer.read(field_name, savepoint)
            else:
                fields_absent.append(field_name)
        [print(f"field  {f} not present in savepoint") for f in fields_absent]
        [print(f"metadata  {f} not present in savepoint") for f in meta_absent]
        return meta_present, fields_present





