from __future__ import division, print_function

from abc import ABC, abstractmethod
from ..base import SerializationMixin


class BaseQAlgorithm(ABC, SerializationMixin):
    def __init__(self, q, alpha):
        self.q = q
        self.env = q.env
        self.alpha = alpha  # this is a property

    @property
    def alpha(self):
        return self.q.alpha

    @alpha.setter
    def alpha(self, alpha):
        self.q.alpha = alpha

    @abstractmethod
    def _delta(self, s, a, r, s_next, a_next=None):
        pass

    def update(self, s, a, r, s_next, a_next=None):
        delta = self._delta(s, a, r, s_next, a_next=a_next)
        self.q._update(s, a, delta)


class BaseVAlgorithm(BaseQAlgorithm):
    @abstractmethod
    def _delta(self, s, r, s_next):
        pass

    def update(self, s, r, s_next):
        delta = self._delta(s, r, s_next)
        self.q._update(s, delta)
