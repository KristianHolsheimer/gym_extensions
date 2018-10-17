from .base import BaseQAlgorithm, BaseVAlgorithm
from ..policies import PolicyQ


class ValueTD0(BaseVAlgorithm):
    def __init__(self, v, alpha=0.01, gamma=0.99):
        self.v = v
        self.alpha = alpha
        self.gamma = gamma

    @property
    def q(self):
        return self.v

    def _delta(self, s, r, s_next):
        v_target = r + self.gamma * self.v(s_next)
        v_pred = self.v(s)
        return v_pred - v_target


class Sarsa(BaseQAlgorithm):
    def __init__(self, q, alpha=0.01, gamma=0.99):
        self.q = q
        self.alpha = alpha
        self.gamma = gamma

    def _delta(self, s, a, r, s_next, a_next):
        q_target = r + self.gamma * self.q(s_next, a_next)
        q_pred = self.q(s, a)
        return q_pred - q_target

    def update(self, s, a, r, s_next, a_next):
        return super(Sarsa, self).update(s, a, r, s_next, a_next)


class QLearning(BaseQAlgorithm):
    def __init__(self, q, alpha=0.01, gamma=0.99):
        self.q = q
        self.policy = PolicyQ(q)
        self.alpha = alpha
        self.gamma = gamma

    def _delta(self, s, a, r, s_next, a_next=None):
        # input a_next is ignored, just leave it unspecified
        a_next = self.policy.greedy(s_next)
        q_target = r + self.gamma * self.q(s_next, a_next)
        q_pred = self.q(s, a)
        return q_pred - q_target
