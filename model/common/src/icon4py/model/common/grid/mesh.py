from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from gt4py.next.common import Dimension

from icon4py.model.common.dimension import (
    CellDim,
    EdgeDim,
    KDim,
    VertexDim,
)
from icon4py.model.common.grid.icon_grid import GridConfig


class BaseMesh(ABC):
    def __init__(self):
        self.config: GridConfig = None
        self.connectivities: Dict[str, np.ndarray] = {}
        self.size: Dict[Dimension, int] = {}

    @property
    @abstractmethod
    def n_cells(self) -> int:
        pass

    @property
    @abstractmethod
    def n_vertices(self) -> int:
        pass

    @property
    @abstractmethod
    def n_edges(self) -> int:
        pass

    @abstractmethod
    def get_offset_provider(self) -> dict:
        pass

    def with_connectivities(self, connectivity: Dict[Dimension, np.ndarray]):
        self.connectivities.update(
            {d.value.lower(): k.astype(int) for d, k in connectivity.items()}
        )
        self.size.update({d: t.shape[1] for d, t in connectivity.items()})

    def with_config(self, config: GridConfig):
        self.config = config
        self._update_size(config)

    def _update_size(self, config: GridConfig):
        self.size[VertexDim] = config.n_vertices
        self.size[CellDim] = config.n_cells
        self.size[EdgeDim] = config.n_edges
        self.size[KDim] = config.k_levels
