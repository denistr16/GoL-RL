import torch
import numpy as np
from model.a2c import ActorCritic


def flatten_grid(grid):
    print (grid.shape)
    return torch.tensor(grid).float().flatten()


class BotPlayer:
    def __init__(self, env, model_path=None, marker=2, max_points_per_step=15):

        self.env = env.get_grid()
        self.model = ActorCritic(self.env.shape[0] * self.env.shape[1],
                                 self.env.shape[0] * self.env.shape[1])
        if model_path is not None:
            snapshot = torch.load(model_path)
            self.model.load_state_dict(snapshot)

        self.marker = marker
        self.max_points_per_step = max_points_per_step

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
