from env.env_2players_naive_torus import NaiveSandbox
from games.gol_with_bot import GameOfLife

from games.players import BotPlayer, HumanPlayer

env = NaiveSandbox((10, 10))


if __name__ == "__main__":
    #GameOfLife(env)
    bot = BotPlayer(env, model_path='/home/artem/Development/research/GoL-RL/snapshots/agent_90.pth')
    human = HumanPlayer('artem', env)

    GameOfLife(env, human, bot)
