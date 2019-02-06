import tkinter as Tkinter
from PIL import Image, ImageTk
import numpy as np
from env.env_2players_naive_torus import NaiveSandbox, players_cells_values, dead_cell

env = NaiveSandbox((30, 30))

H, W = 1000, 1000


def get_cell_id(x, y, grid):
    return int(x / (H / grid.shape[0])), int(y / (W / grid.shape[1]))


class GameOfLife():
    def __init__(self, env):
        self.root = Tkinter.Tk()
        self.frame = Tkinter.Frame(self.root, width=W, height=H)
        self.frame.pack()
        self.canvas = Tkinter.Canvas(self.frame, width=W, height=H)
        self.canvas.place(x=-2, y=-2)

        self.env = env
        self.draw_matrix()

        self.canvas.bind("<Button-1>", self.click_player_one)
        self.canvas.bind("<Button-2>", self.click_player_two)
        # self.canvas.bind('<Key>', self.step)
        self.canvas.bind_all('<Key>', self.step)
        # self.canvas.bind("<Button-3>", self.step)
        self.root.update()
        self.root.mainloop()

    def grid_to_viz(self, grid):
        def to_colors(x):
            if x == 0:
                return 0
            elif x == players_cells_values['player_1']:
                return 255
            else:
                return 125

        return np.vectorize(to_colors)(grid)

    def step(self, event):
        self.env.step()
        self.draw_matrix()

    def draw_matrix(self):
        data = self.env.get_grid()
        data = self.grid_to_viz(data)
        self.im = Image.frombytes('L', (data.shape[1], data.shape[0]), data.astype('b').tostring())
        self.im = self.im.resize((H, W))
        self.photo = ImageTk.PhotoImage(image=self.im)
        self.canvas.create_image(0, 0, image=self.photo, anchor=Tkinter.NW)

    def click_player_one(self, event):
        grid = self.env.get_grid()
        y, x = get_cell_id(event.x, event.y, grid)

        if grid[x][y] == players_cells_values['player_1']:
            grid[x][y] = dead_cell
        elif grid[x][y] == dead_cell:
            grid[x][y] = players_cells_values['player_1']
        self.draw_matrix()

    def click_player_two(self, event):
        grid = self.env.get_grid()
        y, x = get_cell_id(event.x, event.y, grid)

        if grid[x][y] == players_cells_values['player_2']:
            grid[x][y] = dead_cell
        elif grid[x][y] == dead_cell:
            grid[x][y] = players_cells_values['player_2']
        self.draw_matrix()


if __name__ == "__main__":
    GameOfLife(env)
