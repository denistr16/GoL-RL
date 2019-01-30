from abc import ABCMeta, abstractmethod


class IEnv:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "0.1"

    @abstractmethod
    def insert_block(self, block, x0, y0): raise NotImplementedError

    @abstractmethod
    def step(self, n_steps=1): raise NotImplementedError

    @abstractmethod
    def forward(self, inserted_block, inserted_block_position_x0: int, inserted_block_position_y0: int, reward_fn,
                n_steps=1):
        raise NotImplementedError
