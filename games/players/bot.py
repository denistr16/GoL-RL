import torch
import numpy as np
from model.a2c import ActorCritic


def flatten_grid(grid):
    return torch.tensor(grid).float().flatten()


class BotPlayer:
    def __init__(self, env, model_path=None, marker=2, max_points_per_step=15, perception_field_size=(5, 5), hard_x_y=None):

        self.env = np.zeros(env.get_grid().shape)
        self.grid_size = self.env.shape
        self.perception_field_size = perception_field_size
        self.hard_x_y = hard_x_y
        self.model = ActorCritic(self.env.shape[0] * self.env.shape[1],
                                 self.perception_field_size[0]**2+2)

        if model_path is not None:
            snapshot = torch.load(model_path)
            self.model.load_state_dict(snapshot)

        self.marker = marker
        self.max_points_per_step = max_points_per_step

    def insert_block(self, block, x0, y0):
        self.env[x0: x0 + block.shape[0], y0: y0 + block.shape[1]] = block

    def reset(self):
        self.env *= 0

    def probs_to_cells(self, probs, topk=3, marker=2):
        probs_top_k, idx_top_k = probs.topk(topk)
        inserted_block = np.zeros(self.perception_field_size)
        inserted_block = inserted_block.flatten()
        inserted_block[idx_top_k] = marker
        return inserted_block.reshape(self.perception_field_size)

    def convert_coordinates(self,x , y):
        x, y = torch.round(x * (self.grid_size[0] - self.perception_field_size[0])), torch.round(
            y * (self.grid_size[0] - self.perception_field_size[0]))
        return x.detach().int(), y.detach().int()

    def step(self, env):
        self.reset()
        probs = self.model.get_action_probs(flatten_grid(env)).detach()
        perception_field, x, y = probs[:-2], probs[-1], probs[-2]
        x, y = self.convert_coordinates(x, y)
        perception_field = self.probs_to_cells(probs=perception_field,
                                            topk=self.max_points_per_step,
                                            marker=self.marker)
        if self.hard_x_y is not None:
            x, y = self.hard_x_y
        self.insert_block(perception_field, x, y)
        return self.env
