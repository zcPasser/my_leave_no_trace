import numpy as np
import random
# import t
from collections import deque

class MemoryBuffer:
    def __init__(self, size):
        self.bufer = deque(maxlen=size)
        self.max_size = size
        self.len_ = 0

    def sample(self, batch_size):
        """
        samples a random batch from the replay memory buffer
        :param batch_size:
        :return: batch (numpy array)
        """
        batch = []
        batch_size = min(batch_size, self.len_)
        # sample(list, k)返回一个长度为k新列表，新列表存放list所产生k个随机唯一的元素
        batch = random.sample(self.bufer, batch_size)

        # experience:(s, a, r, s_) 对应所有数据 进行 转换float32
        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s__arr = np.float32([arr[3] for arr in batch])

        return s_arr, a_arr, r_arr, s__arr

    def len(self):
        return self.len_

    def add(self, s, a, r, s_):
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s_: next state
        :return:
        """
        experience = (s, a, r, s_)
        self.len_ += 1
        if self.len_ > self.max_size:
            self.len_ = self.max_size
        self.bufer.append(experience)
