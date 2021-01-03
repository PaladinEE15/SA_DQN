import warnings
from typing import Generator, Optional, Union

import numpy as np
import torch as th
from gym import spaces


class StateBuffer(object):

    def __init__(
        self,
        d_type,
        buffer_size: int,
        device: Union[th.device, str] = "cuda",
    ):
        super(StateBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.observations = np.zeros((self.buffer_size, ) + self.obs_shape, dtype=d_type)

    def add(self, obs:np.ndarray) -> None:
        self.observations[self.pos] = np.array(obs).copy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)  
        return th.as_tensor(self.observations[batch_inds,:]).to(self.device)
              

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False    