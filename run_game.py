from env.env_2players_naive_torus import NaiveSandbox
from games.gol_with_bot import GameOfLife

from games.players.bot import BotPlayer
from games.players.human import HumanPlayer

env = NaiveSandbox((20, 20))


if __name__ == "__main__":
    bot = BotPlayer(model_path='./snapshots/agent_90.pth', env=env, marker=2, max_points_per_step=9, perception_field_size=(4, 4))
    human = HumanPlayer('artem', env)

    GameOfLife(env, human, bot, steps_after_action=20)
