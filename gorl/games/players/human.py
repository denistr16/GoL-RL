from gorl.env.env_2players_naive_torus import dead_cell

## H, W - for height and width of game window area.
## Not number of cells, but image size in pixels.
def get_cell_id(x, y, grid, H=1000, W=1000):
    return int(x / (H / grid.shape[0])), int(y / (W / grid.shape[1]))

class HumanPlayer:
    def __init__(self, name, env, marker=1, max_points_per_step=15):
        self.name = name
        self.grid = env.get_grid().copy()
        self.max_points_per_step = max_points_per_step
        self.points_left = max_points_per_step
        self.marker = marker

    def step(self, event):
        y, x = get_cell_id(event.x, event.y, self.grid)
        if self.points_left == 0:
            print ("No cells left for this round")
            return self.grid
        if self.grid[x][y] == self.marker:
            self.grid[x][y] = dead_cell
        elif self.grid[x][y] == dead_cell:
            self.grid[x][y] = self.marker
        self.points_left -= 1
        print ("Cells left: ", self.points_left)
        return self.grid

    def reset(self, env):
        self.points_left = self.max_points_per_step
        self.grid = env.get_grid().copy()
