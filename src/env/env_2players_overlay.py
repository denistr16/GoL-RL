import random

from .interface_env import IEnv

template_rewards = {
    "alive_cells": AliveCellsReward()
}

players_cells_values = {'player_1': 1, 'player_2': 3}
players_overlay_values = {'player_1': 2, 'player_2': 4}

sv = set(players_cells_values.values()) & set(players_overlay_values.values())
assert len(sv) == 0, "players_cells_values and players_overlay_values contain the same key values: {}".format(sv)

alive_cell_values = set(players_cells_values.values())
dead_cell = 0


class NaiveSandbox(IEnv):
    def __init__(self, grid_size, n_iterations=None):
        self.__dead_cell = dead_cell
        self.__alive_cell = alive_cell_values

        self.__grid = np.full(grid_size, self.__dead_cell)

        self.__n_iterations = n_iterations
        self.__current_iteration = 0

    def insert_block(self, block, x0, y0):
        block_shape = block.shape

        # check if the new block is inside of the grid
        # if not - return without insert
        if x0 + block_shape[0] > self.__grid.shape[0] or y0 + block_shape[1] > self.__grid.shape[1]:
            return

        self.__grid[x0: x0 + block_shape[0], y0: y0 + block_shape[1]] = block

    def __count_neighbors(self, i, j, g_len, value):
        neighbors_value = [self.__grid[i, (j - 1) % g_len],
                           self.__grid[i, (j + 1) % g_len],
                           self.__grid[(i - 1) % g_len, j],
                           self.__grid[(i + 1) % g_len, j],
                           self.__grid[(i - 1) % g_len, (j - 1) % g_len],
                           self.__grid[(i - 1) % g_len, (j + 1) % g_len],
                           self.__grid[(i + 1) % g_len, (j - 1) % g_len],
                           self.__grid[(i + 1) % g_len, (j + 1) % g_len]]
        return sum([k == value for k in neighbors_value])

    def step(self, n_steps=1):
        for i in range(n_steps):
            new_grid = self.__grid.copy()
            g_len = len(new_grid)

            for i in range(g_len):
                for j in range(g_len):
                    p1_total = self.__count_neighbors(i, j, g_len, value=players_cells_values['player_1'])
                    p2_total = self.__count_neighbors(i, j, g_len, value=players_cells_values['player_2'])
                    total = p1_total + p2_total

                    if self.__grid[i, j] in self.__alive_cell:
                        if (total < 2) or (total > 3):
                            new_grid[i, j] = self.__dead_cell
                    else:
                        if total == 3:
                            if p1_total > p2_total:
                                new_grid[i, j] = players_cells_values['player_1']
                            elif p2_total > p1_total:
                                new_grid[i, j] = players_cells_values['player_2']
                            else:
                                new_grid[i, j] = random.choice(list(self.__alive_cell))

            self.__grid[:] = new_grid[:]
            self.__current_iteration += 1

    def is_done(self):
        if self.__n_iterations is None:
            return False
        elif self.__current_iteration >= self.__n_iterations:
            return True
        else:
            return False

    def forward(self, inserted_block, inserted_block_position_x0: int, inserted_block_position_y0: int,
                reward_fn=template_rewards['alive_cells'], n_steps=1):
        self.insert_block(block=inserted_block, x0=inserted_block_position_x0, y0=inserted_block_position_y0)
        self.step(n_steps=n_steps)
        reward_value = reward_fn(self.__grid)
        is_done = self.is_done()
        env_state = {"reward": reward_value, "grid": self.__grid, 'done': is_done}
        return env_state

    def get_grid(self):
        return self.__grid
