import numpy as np


class AliveCellsReward:
    def __call__(self, env):
        return self.forward(env)

    def forward(self, env):
        return env.sum() / (env.shape[0] * env.shape[1])


class MultipleAgentsCellsReward:
    def __call__(self, env):
        return self.forward(env)

    def forward(self, env):
        env = env.flatten()
        cell_count = np.bincount(env, minlength=3)
        rewards = [i / sum(cell_count) for i in cell_count[1:]]
        return rewards


class MultipleAgentsCellsRewardWithEnemyDiscount:
    def __call__(self, env):
        return self.forward(env)

    def forward(self, env):
        env = env.flatten()
        dead_cells, first_player, second_player = np.bincount(env, minlength=3)
        first_reward = first_player * (first_player - second_player) / dead_cells
        second_reward = second_player * (second_player - first_player) / dead_cells
        return [first_reward, second_reward]
