from reward.rewards import *
from .interface_env import IEnv

template_rewards = {
    "alive_cells": AliveCellsReward()
}

players_cells_values = {'player_1': 1, 'player_2': 2}
alive_cell_values = set(players_cells_values.values())
dead_cell = 0


class NaiveSandbox(IEnv):
    def __init__(self, grid_size, n_iterations=None):
        self.__dead_cell = dead_cell
        self.__alive_cell = alive_cell_values

        self.__grid = np.full(grid_size, self.__dead_cell, dtype=np.int8)
        # self.__grid = np.random.randint(0, 3, size=shape, dtype=dtype)

        self.__neighbor = np.zeros(self.__grid.shape, dtype=np.int8)
        self.__neighbor_2 = np.zeros(self.__grid.shape, dtype=np.int8)
        self.__neighbor_id = self.__make_neighbor_indices()

        self.__n_iterations = n_iterations
        self.__current_iteration = 0

    @staticmethod
    def __make_neighbor_indices():
        d = [slice(None), slice(1, None), slice(0, -1)]
        d2 = [
            (0, 1), (1, 1), (1, 0), (1, -1)
        ]
        out = [None for i in range(8)]
        for i, idx in enumerate(d2):
            x, y = idx
            out[i] = [d[x], d[y]]
            out[7 - i] = [d[-x], d[-y]]
        return out

    def __count_neighbors(self):
        self.__neighbor[:, :] = 0
        self.__neighbor_2[:, :] = 0

        w = self.__grid
        n_id = self.__neighbor_id

        n = self.__neighbor
        n2 = self.__neighbor_2
        w1 = w & 1
        w2 = w & 2
        w2 = w2 >> 1

        for i in range(8):
            n[n_id[i]] += w1[n_id[7 - i]]
            n2[n_id[i]] += w2[n_id[7 - i]]

    def __update_grid(self):
        w = self.__grid
        n = self.__neighbor
        n2 = self.__neighbor_2

        w1 = w & 1
        w2 = w & 2
        w2 = w2 >> 1

        w1 &= (((n == 2) | (n == 3)) & (n2 <= 3))
        w2 &= ((n <= 3) & ((n2 == 2) | (n2 == 3)))

        w1 |= ((n == 3) & (n2 != 3))
        w2 |= ((n2 == 3) & (n != 3))

        w2 = w2 << 1
        self.__grid = w1 | w2

    def step(self, n_steps=1):
        self.__count_neighbors()
        self.__update_grid()
        self.__current_iteration += 1

    def is_done(self):
        if self.__n_iterations is None:
            return False
        elif self.__current_iteration >= self.__n_iterations:
            return True
        else:
            return False

    def insert_block(self, block, x0, y0):
        block_shape = block.shape
        # check if the new block is inside of the grid
        # if not - return without insert
        if x0 + block_shape[0] > self.__grid.shape[0] or y0 + block_shape[1] > self.__grid.shape[1]:
            return

        self.__grid[x0: x0 + block_shape[0], y0: y0 + block_shape[1]] = block

    def forward(self, inserted_block, inserted_block_position_x0: int, inserted_block_position_y0: int,
                reward_fn=template_rewards['alive_cells'], n_steps=1):

        self.insert_block(block=inserted_block, x0=inserted_block_position_x0, y0=inserted_block_position_y0)

        is_done = False
        for i in range(n_steps):
            self.step(n_steps=n_steps)
            is_done = self.is_done()
            if is_done:
                break

        reward_value = reward_fn(self.__grid)
        env_state = {"reward": reward_value, "grid": self.__grid, 'done': is_done}

        return env_state

    def get_grid(self):
        return self.__grid
