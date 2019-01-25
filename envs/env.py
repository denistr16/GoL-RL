import torch
import numpy as np


class Env:
    def __init__(self, grid_size=(100, 100)):
        self.grid = np.array(grid_size)

    def step(self, perception_field: torch.tensor, field_position_x: int, field_position_y: int):
        
        return {"grid": self.grid}
