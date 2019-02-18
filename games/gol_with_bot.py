import tkinter as Tkinter
from PIL import Image, ImageTk
import numpy as np
from time import sleep
from env.env_2players_naive_torus import players_cells_values, dead_cell

"""
### RULES ###
1. Each player can place up to 15 cells per round.
2. Players don't know how they placed cells.
3. If two players place cell at same location - no cell is created at all.
4. Round longs 10 environment steps. By the end of the round, wins player with alive cells.
5. If both player cells are alive, it's draw.
"""

H, W = 1000, 1000

def merge_agents(first_values,second_values):
    if all([first_values, second_values]):
        return 0
    return second_values if first_values == 0 else first_values

def merge_perceptions(old, new):
    return new if old == 0 else old

class GameOfLife():
    def __init__(self, env, player1, player2, steps_after_action=10):
        self.root = Tkinter.Tk()
        self.frame = Tkinter.Frame(self.root, width=W, height=H)
        self.frame.pack()
        self.canvas = Tkinter.Canvas(self.frame, width=W, height=H)
        self.canvas.place(x=-2, y=-2)

        self.env = env
        self.render()

        self.player1 = player1
        self.player2 = player2
        self.steps_after_action = steps_after_action
        self.canvas.bind("<Button-1>", self.human_step)
        self.canvas.bind_all('<space>', self.round)
        self.root.update()
        self.root.mainloop()

    def round(self, event):
        print ("Bot's Move")
        player1_moves = self.player1.grid
        player2_moves = self.player2.step(self.env.get_grid())
        all_moves = np.vectorize(merge_agents)(player1_moves, player2_moves)
        new_grid_state = np.vectorize(merge_perceptions)(self.env.get_grid(), all_moves)
        self.env.insert_block(new_grid_state, 0, 0)
        self.render()
        sleep(2)
        for i in range(self.steps_after_action):
           self.step(None)
        self.player1.reset(self.env)
        print ("Player's Move")

    def human_step(self, event):
        grid = self.player1.step(event)
        self.render(grid)

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
        self.render()

    def render(self, grid=None):
        if grid is None:
            data = self.env.get_grid()
        else:
            data = grid
        data = self.grid_to_viz(data)
        self.im = Image.frombytes('L', (data.shape[1], data.shape[0]), data.astype('b').tostring())
        self.im = self.im.resize((H, W))
        self.photo = ImageTk.PhotoImage(image=self.im)
        self.canvas.create_image(0, 0, image=self.photo, anchor=Tkinter.NW)
        self.root.update()
