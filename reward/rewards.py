import numpy as np
import torch

class AliveCellsReward:

    def __call__(self, env):
        return self.forward(env)

    def forward(self, env):
        return env.sum() / (env.shape[0]*env.shape[1])
