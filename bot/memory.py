import numpy as np
from random import sample

class ReplayMemory:
    def __init__(self, capacity, resolution):
        state_shape = (capacity, resolution[0], resolution[1], 3)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.bool_)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, s2, action, isterminal, reward):
        self.s1[self.pos] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        print("samples shape", self.s1[i].shape)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]