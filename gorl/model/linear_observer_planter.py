import torch


class LinearObserever(torch.nn.Module):

    def __init__(self, name, grid_size: tuple, window_size: int):
        super().__init__()
        self.name = name
        self.grid_size = grid_size
        self.window_size = (window_size, window_size)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(grid_size[0] * grid_size[1], 2),
            torch.nn.ReLU())

    def forward(self, observations):
        x, y = self.model.forward(observations.flatten())
        x, y = torch.round(x * (self.grid_size[0] - self.window_size[0])), torch.round(
            y * (self.grid_size[0] - self.window_size[0]))
        return x, y


class LinearPlanter(torch.nn.Module):
    def __init__(self, name, window_size: int):
        super().__init__()
        self.name = name
        self.window_size = window_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(window_size * window_size, window_size * window_size),
            torch.nn.Sigmoid()
        )

    def forward(self, observations):
        new_points = torch.round(self.model.forward(observations.flatten()))
        return new_points.view(observations.shape[0], observations.shape[1])


class LinearPlanterObserver(torch.nn.Module):
    def __init__(self, grid_size: tuple, window_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.window_size = window_size

        self.memoized_env = torch.nn.Parameter(torch.zeros(grid_size[0], grid_size[1]), requires_grad=False)
        self.planter = LinearPlanter("Sam", window_size)
        self.observer = LinearObserever("Mike", grid_size, window_size)

    def forward(self, env):
        x, y = self.observer.forward(env)

        x0, x1, y0, y1 = [x.detach().int(), x.detach().int() + self.window_size,
                          y.detach().int(), y.detach().int() + self.window_size]

        perception_field = self.planter.forward(env[x0:x1, y0:y1])
        memoized_env = env * self.memoized_env
        memoized_env[x0:x1, y0:y1] = perception_field

        return memoized_env, perception_field, x0, y0
