from gorl.env.env_2players_naive_torus import NaiveSandbox
from gorl.games.gol_with_bot import GameOfLife

from gorl.games.players.bot import BotPlayer
from gorl.games.players.human import HumanPlayer

env = NaiveSandbox((20, 20))


if __name__ == "__main__":
    bot = BotPlayer(env=env, marker=2, max_points_per_step=9,
                    perception_field_size=(4, 4))
    human = HumanPlayer('artem', env)

    GameOfLife(env, human, bot, steps_after_action=20)
