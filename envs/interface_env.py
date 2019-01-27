from abc import ABCMeta, abstractmethod


# This call will fail with an exception
# try:
#     x = MyClient(MyBadServer)
# except Exception as exc:
#     print 'Failed as it should!'
#
# # This will pass with glory
# MyClient(MyServer()).client_show()


class IEnv:
    __metaclass__ = ABCMeta

    @classmethod
    def version(cls): return "0.1"

    @abstractmethod
    def insert_block(self, block, x0, y0): raise NotImplementedError

    @abstractmethod
    def step(self, n_steps=1): raise NotImplementedError

    @abstractmethod
    def forward(self, inserted_block, inserted_block_position_x0: int, inserted_block_position_y0: int, n_steps=1):
        raise NotImplementedError
