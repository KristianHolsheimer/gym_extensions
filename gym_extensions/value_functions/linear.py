from __future__ import print_function, division

import numpy as np

from .base import BaseQ
from ..utils import feature_vector


class LinearQ(BaseQ):
    """State-action value function implementation using linear-model approximation.

    Args:
        env (gym-environment): This is needed to get the dimensions of the state-action space.
        alpha (float): learning rate; may be overwritten by Algorithm class
        intercept (bool): Whether to add include an intercept in the set of weights.
        polynomial_degree (int): The degree for polynomial interactions.

    """
    def __init__(self, env, alpha=0.1, intercept=True, polynomial_degree=1, l1=0.0, l2=0.0):
        super(LinearQ, self).__init__(env, alpha)
        self.alpha = alpha
        self.intercept = intercept
        self.polynomial_degree = polynomial_degree
        self.l1 = l1
        self.l2 = l2

        self.init_weights()

    def __call__(self, s, a):
        x = self._feature_vector(s, a)
        return np.dot(x, self.w)

    def init_weights(self):
        s = self.env.observation_space.sample()
        a = self.env.action_space.sample()
        x = self._feature_vector(s, a)
        self.w = np.zeros_like(x)
        self.v = np.zeros_like(self.w)

    def _feature_vector(self, s, a):
        x = feature_vector(self.env, s, a, intercept=self.intercept, polynomial_degree=self.polynomial_degree)
        return x

    def _grad(self, s, a, delta):
        return delta * self._feature_vector(s, a)

    def _update(self, s, a, delta):
        """we use basic gradient descent with momentum for our weight updates"""

        if np.isnan(self.w).any():
            raise RuntimeError("weight vector contains NaN's")

        # compute gradient
        g = self._grad(s, a, delta) + self.l2 * self.w + self.l1 * np.sign(self.w)

        # update momentum and weights
        self.v = 0.9 * self.v + 0.1 * self.alpha * g
        self.w = self.w - self.v


class LinearV(LinearQ):
    def __call__(self, s):
        x = self._feature_vector(s, None)
        return np.dot(x, self.w)

    def init_weights(self):
        s = self.env.observation_space.sample()
        x = self._feature_vector(s, None)
        self.w = np.zeros_like(x)
        self.v = np.zeros_like(self.w)

    def _update(self, s, delta):
        return super(self.__class__, self)._update(s, None, delta)
