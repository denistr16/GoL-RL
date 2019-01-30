from .interface_env import IEnv
import numpy as np


class NaiveSandbox(IEnv):
    def __init__(self, grid_size):
        self.__dead_cell = 0
        self.__alive_cell = 1
        self.__grid = np.full(grid_size, self.__dead_cell)

    def insert_block(self, block, x0, y0):
        block_shape = block.shape

        # check if the new block is inside of the grid
        # if not - return without insert
        if x0 + block_shape[0] > self.__grid.shape[0] or y0 + block_shape[1] > self.__grid.shape[1]:
            return

        self.__grid[x0: x0 + block_shape[0], y0: y0 + block_shape[1]] = block

    def step(self, n_steps=1):
        for i in range(n_steps):
            new_grid = self.__grid.copy()
            g_len = len(new_grid)

            for i in range(g_len):
                for j in range(g_len):

                    # compute the 8-neighbor sum
                    # using toroidal boundary conditions - x and y wrap around
                    # so that the simulation takes place on a torus's surface.
                    total = int((self.__grid[i, (j - 1) % g_len] + self.__grid[i, (j + 1) % g_len] +
                                 self.__grid[(i - 1) % g_len, j] + self.__grid[(i + 1) % g_len, j] +
                                 self.__grid[(i - 1) % g_len, (j - 1) % g_len] +
                                 self.__grid[(i - 1) % g_len, (j + 1) % g_len] +
                                 self.__grid[(i + 1) % g_len, (j - 1) % g_len] +
                                 self.__grid[(i + 1) % g_len, (j + 1) % g_len])
                                / self.__alive_cell)

                    # apply cell rules
                    if self.__grid[i, j] == self.__alive_cell:
                        if (total < 2) or (total > 3):
                            new_grid[i, j] = self.__dead_cell
                    else:
                        if total == 3:
                            new_grid[i, j] = self.__alive_cell

            self.__grid[:] = new_grid[:]

    def forward(self, inserted_block, inserted_block_position_x0: int, inserted_block_position_y0: int, reward_fn,
                n_steps=1):
        self.insert_block(block=inserted_block.numpy(), x0=inserted_block_position_x0, y0=inserted_block_position_y0)
        self.step(n_steps=n_steps)
        reward_value = reward_fn(self.__grid)

        env_state = {"reward": reward_value, "grid": self.__grid}
        return env_state

    def get_grid(self):
        return self.__grid
