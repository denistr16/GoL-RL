import torch
import numpy as np


class Env:
    def __init__(self, grid_size=(100, 100)):
        self.__dead_cell = 0
        self.__alive_cell = 1

        self.grid = np.full(grid_size, self.__dead_cell)

    def step(self, perception_field: torch.tensor, field_position_x: int, field_position_y: int):
        
        return {"grid": self.grid}
