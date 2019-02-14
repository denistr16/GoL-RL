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
