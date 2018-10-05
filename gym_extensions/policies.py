import numpy as np
from gym.spaces import Discrete

from .base import SerializationMixin
from .utils import softmax


class PolicyQ(SerializationMixin):
    def __init__(self, q, epsilon=0.01, seed=None):
        self.q = q
        self.epsilon = epsilon
        self.seed = seed

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, new_seed):
        self._seed = new_seed
        self.random = np.random.RandomState(new_seed)

    def greedy(self, s):
        if isinstance(self.q.env.action_space, Discrete):
            # here we ensure that ties are broken randomly
            actions = np.arange(self.q.env.action_space.n)
            self.random.shuffle(actions)
            i = np.argmax([self.q(s, a) for a in actions])
            a = actions[i]
            return a
        else:
            raise NotImplementedError("Haven't yet implemented action space: {}".format(self.q.env.action_space))

    def proba(self, s):
        if isinstance(self.q.env.action_space, Discrete):
            # of course, q values aren't exactly probabilities, but whatevs
            proba = softmax([self.q(s, a) for a in range(self.q.env.action_space.n)])
            return proba
        else:
            raise NotImplementedError("Haven't yet implemented action space: {}".format(self.q.env.action_space))

    def random_uniform(self, s):
        """We keep s as a (trivial) argument to keep the possibility open that
        the action space is state dependent.
        """
        return self.q.env.action_space.sample()

    def random_thompson(self, s):
        p = self.proba(s)
        n = self.q.env.action_space.n
        a = self.random.choice(n, p=p)
        return a

    def epsilon_greedy(self, s):
        if self.random.rand() < self.epsilon:
            return self.random_uniform(s)
        else:
            return self.greedy(s)
