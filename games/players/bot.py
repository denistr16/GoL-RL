import torch
import numpy as np
from model.a2c import ActorCritic


def flatten_grid(grid):
    print (grid.shape)
    return torch.tensor(grid).float().flatten()


class BotPlayer:
    def __init__(self, env, model_path=None, marker=2, max_points_per_step=15, perception_field_size=(5, 5)):

        self.env = env.copy()

        if model_path is not None:
            snapshot = torch.load(model_path)
            self.model.load_state_dict(snapshot)

        self.marker = marker
        self.max_points_per_step = max_points_per_step
        self.perception_field_size = perception_field_size

        self.model = ActorCritic(self.env.get_grid().shape[0] * self.env.get_grid().shape[1],
                                 self.perception_field_size[0]**2)

    def probs_to_cells(self, probs, env, topk=3, marker=1):
        probs_top_k, idx_top_k = probs.topk(topk)
        inserted_block = np.zeros(self.perception_field_size)
        inserted_block = inserted_block.flatten()
        inserted_block[idx_top_k] = marker
        return inserted_block.reshape(self.perception_field_size)

    def step(self, env):
        probs = self.model.get_action_probs(flatten_grid(env))
        perception_field = self.probs_to_cells(probs=probs.detach(),
                                            env=self.env.get_grid(),
                                            topk=self.max_points_per_step,
                                            player=self.marker)
        return perception_field
