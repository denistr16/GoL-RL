from env.env_2players_naive_torus import players_cells_values, dead_cell


H, W = 1000, 1000

def get_cell_id(x, y, grid):
    return int(x / (H / grid.shape[0])), int(y / (W / grid.shape[1]))


class HumanPlayer:
    def __init__(self, name, env):
        self.name = name
        self.grid = env.get_grid().copy()
        self.max_points_per_step = 15
        self.points_left = 15

    def step(self, event):
        y, x = get_cell_id(event.x, event.y, self.grid)
        if self.points_left == 0:
            print ("No cells left for this round")
            return self.grid
        if self.grid[x][y] == players_cells_values['player_1']:
            self.grid[x][y] = dead_cell
        elif self.grid[x][y] == dead_cell:
            self.grid[x][y] = players_cells_values['player_1']
        self.points_left -= 1
        print ("Cells left: ", self.points_left)
        return self.grid

    def reset(self, env):
        self.points_left = self.max_points_per_step
        self.grid = env.get_grid().copy()


def flatten_grid(grid):
    print (grid.shape)
    return torch.tensor(grid).float().flatten()

import torch
import numpy as np
from model.a2c import ActorCritic
class BotPlayer:
    def __init__(self, env, model_path=None):

        self.env = env.get_grid()
        self.model = ActorCritic(self.env.shape[0] * self.env.shape[1],
                                 self.env.shape[0] * self.env.shape[1])
        if model_path is not None:
            snapshot = torch.load(model_path)
            self.model.load_state_dict(snapshot)

        self.marker = 2
        self.max_points_per_step = 15

    def probs_to_cells(self, probs, env, topk=3, player=1):
        probs_top_k, idx_top_k = probs.topk(topk)
        inserted_block = np.zeros(env.shape)
        inserted_block = inserted_block.flatten()
        inserted_block[idx_top_k] = player
        return inserted_block.reshape(env.shape)

    def step(self):
        probs = self.model.get_action_probs(flatten_grid(self.env))
        perception_field = self.probs_to_cells(probs=probs.detach(),
                                            env=self.env,
                                            topk=self.max_points_per_step,
                                            player=self.marker)
        return perception_field
