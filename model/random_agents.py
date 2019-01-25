import numpy as np
import torch

class RandomObserever(torch.nn.Module):

    def __init__(self, name, grid_size=100, window_size=10):
        super().__init__()
        self.name = name
        self.grid_size = (grid_size, grid_size)
        self.window_size = (window_size, window_size)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(grid_size*grid_size, 2),
            torch.nn.ReLU())

    def forward(self, observations):
        x, y = self.model.forward(observations.flatten())
        x, y = torch.round(x*(self.grid_size[0]-self.window_size[0])),torch.round(y*(self.grid_size[0]-self.window_size[0]))
        return x, y

class RandomPlanter(torch.nn.Module):
    def __init__(self, name, window_size=10):
        super().__init__()
        self.name = name
        self.window_size = window_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(window_size*window_size, window_size*window_size),
            torch.nn.Sigmoid()
            )

    def forward(self, observations):
        new_points = torch.round(self.model.forward(observations.flatten()))
        return new_points.view(observations.shape[0], observations.shape[1])

class ObserverAndPlanter(torch.nn.Module):
    def __init__(self, grid_size=100, window_size=10):
        super().__init__()

        self.observer = RandomObserever("Sam", grid_size, window_size)
        self.planter = RandomPlanter("Mike", window_size)
        self.window_size = window_size
        self.grid_size = grid_size

    def forward(self, env):
        x, y = self.observer.forward(env)
        x0, x1, y0, y1 = [x.detach().int(),x.detach().int()+self.window_size, y.detach().int(),y.detach().int()+self.window_size]
        return self.planter.forward(env[x0:x1, y0:y1]), x0, x1, y0, y1
