class AliveCellsReward:
    def __call__(self, env):
        return self.forward(env)

    def forward(self, env):
        return env.sum() / (env.shape[0]*env.shape[1])

import numpy as np
class MultipleAgentsCellsReward:
    def __call__(self, env):
        return self.forward(env)

    def forward(self, env):
        env = env.flatten()
        cell_count = np.bincount(env)
        rewards = [i / sum(cell_count) for i in cell_count[1:]]
        if len(rewards) == 0:
            return [0.0, 0.0]
        return rewards
