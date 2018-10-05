from abc import ABC, abstractmethod
from ..base import SerializationMixin


class BaseQ(ABC, SerializationMixin):
    def __init__(self, env, alpha):
        self.env = env
        self.alpha = alpha

    @abstractmethod
    def __call__(self, s, a):
        """Abstract method for calling a Q-function; it returns the estimated
        value of a state-action pair.
        """

    @abstractmethod
    def _update(self, s, a, residual):
        """Abstract method for updating a Q-function, given state-action pair and
        the objective residual (e.g. TD-error in case TD learning).
        """
