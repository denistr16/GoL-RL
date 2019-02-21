import matplotlib.pyplot as plt
import numpy as np


class World(object):
    def __init__(self, shape, random=True, dtype=np.int8):
        if random:
            self.data = np.random.randint(0, 3, size=shape, dtype=dtype)
        else:
            self.data = np.zeros(shape, dtype=dtype)
        self.shape = self.data.shape
        self.dtype = dtype
        self._engine = Engine(self)

        self.step = 0

    def animate(self):
        return Animate(self).animate()

    def __str__(self):
        # probably can make a nicer text output here.
        return self.data.__str__()


class Animate(object):
    def __init__(self, world):
        self.world = world
        self.im = None

    def animate(self):
        while True:
            if self.world.step == 0:
                plt.ion()
                self.im = plt.imshow(self.world.data, vmin=0, vmax=2,
                                     cmap=plt.cm.gray)
            else:
                self.im.set_data(self.world.data)

            print('w_step')
            self.world.step += 1
            self.world._engine.next_state()
            plt.pause(0.01)
            yield self.world


class Engine(object):
    def __init__(self, world, dtype=np.int8):
        self._world = world
        self.shape = world.shape
        self.neighbor = np.zeros(world.shape, dtype=dtype)
        self.neighbor_2 = np.zeros(world.shape, dtype=dtype)
        self._neighbor_id = self._make_neighbor_indices()

    def _make_neighbor_indices(self):
        # create a list of 2D indices that represents the neighbors of each
        # cell such that list[i] and list[7-i] represents the neighbor at
        # opposite directions. The neighbors are at North, NE, E, SE, S, SW,
        # W, NE directions.
        d = [slice(None), slice(1, None), slice(0, -1)]
        d2 = [
            (0, 1), (1, 1), (1, 0), (1, -1)
        ]
        out = [None for i in range(8)]
        for i, idx in enumerate(d2):
            x, y = idx
            out[i] = [d[x], d[y]]
            out[7 - i] = [d[-x], d[-y]]
        return out

    def _count_neighbors(self):
        self.neighbor[:, :] = 0  # reset neighbors
        self.neighbor_2[:, :] = 0

        # count #neighbors of each cell.
        w = self._world.data
        n_id = self._neighbor_id

        n = self.neighbor
        n2 = self.neighbor_2
        w1 = w & 1
        w2 = w & 2
        w2 = w2 >> 1
        #         print('w1',w1)
        #         print('w2',w2)

        for i in range(8):
            n[n_id[i]] += w1[n_id[7 - i]]
            n2[n_id[i]] += w2[n_id[7 - i]]

        # print('n1',self.neighbor)
        # print('n2', self.neighbor_2)

    def _update_world(self):
        w = self._world.data
        n = self.neighbor
        n2 = self.neighbor_2

        w1 = w & 1
        w2 = w & 2
        w2 = w2 >> 1

        w1 &= (((n == 2) | (n == 3)) & (n2 <= 3))
        w2 &= ((n <= 3) & ((n2 == 2) | (n2 == 3)))

        w1 |= ((n == 3) & (n2 != 3))
        w2 |= ((n2 == 3) & (n != 3))

        w2 = w2 << 1

        self._world.data = w1 | w2

    def next_state(self):
        self._count_neighbors()
        self._update_world()


def main():
    world = World((100, 100))

    for _ in world.animate():
        pass


if __name__ == '__main__':
    main()

