import torch
import numpy as np


class Env:
    def __init__(self, grid_size=(100, 100)):
        self.__dead_cell = 0
        self.__alive_cell = 1

        self.grid = np.full(grid_size, self.__dead_cell)

    def grid_update(self):
        new_grid = self.grid.copy()
        g_len = len(new_grid)

        for i in range(g_len):
            for j in range(g_len):

                # compute 8-neghbor sum
                # using toroidal boundary conditions - x and y wrap around
                # so that the simulaton takes place on a toroidal surface.
                total = int((self.grid[i, (j - 1) % g_len] + self.grid[i, (j + 1) % g_len] +
                             self.grid[(i - 1) % g_len, j] + self.grid[(i + 1) % g_len, j] +
                             self.grid[(i - 1) % g_len, (j - 1) % g_len] + self.grid[(i - 1) % g_len, (j + 1) % g_len] +
                             self.grid[(i + 1) % g_len, (j - 1) % g_len] + self.grid[(i + 1) % g_len, (j + 1) % g_len])
                            / self.__alive_cell)

                # apply Conway's rules
                if self.grid[i, j] == self.__alive_cell:
                    if (total < 2) or (total > 3):
                        new_grid[i, j] = self.__dead_cell
                else:
                    if total == 3:
                        new_grid[i, j] = self.__alive_cell

                        # update data
        self.grid[:] = new_grid[:]

    def insert_block_into_grid(self, block, x, y):
        block_shape = block.shape

        # check if the new block is inside of the grid
        # if not - return without insert
        if x + block_shape[0] > self.grid.shape[0] or y + block_shape[1] > self.grid.shape[1]:
            return

        self.grid[x: x + block_shape[0], y: y + block_shape[1]] = block

    def step(self, perception_field: torch.tensor, field_position_x: int, field_position_y: int):
        block = perception_field.numpy()

        self.insert_block_into_grid(block=block, x=field_position_x, y=field_position_y)
        self.grid_update()

        return {"grid": self.grid}
