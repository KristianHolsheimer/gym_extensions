import numpy as np

from .base import BaseQ
from ..utils import check_observation_action_pair, check_action_space_type, check_observation_space_type
from gym.spaces.discrete import Discrete


class TabularQ(BaseQ):
    def __init__(self, env, alpha=0.1, optimistic_initialization=0):
        check_action_space_type(Discrete, env)
        check_observation_space_type(Discrete, env)

        self.env = env
        self.alpha = alpha
        self.optimistic_initialization = optimistic_initialization
        self.init_table()

    def init_table(self):
        self.table = self.optimistic_initialization * np.ones([self.env.observation_space.n, self.env.action_space.n])

    def __call__(self, s, a):
        check_observation_action_pair(s, a, self.env)
        return self.table[s, a]

    def _update(self, s, a, residual):
        self.table[s, a] -= self.alpha * residual

