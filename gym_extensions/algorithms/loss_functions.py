from __future__ import division, print_function

from abc import ABC, abstractmethod


class BaseLoss(ABC):
    @abstractmethod
    def __call__(self, q_true, q_pred):
        pass


class BaseDifferentiableLoss(ABC, BaseLoss):
    @abstractmethod
    def grad(self, q_true, q_pred):
        pass


class OrdinaryLeastSquaresLoss(BaseDifferentiableLoss):
    def __call__(self, q_true, q_pred):
        err = q_pred - q_true
        return err * err / 2

    def grad(self, q_true, q_pred):
        return q_pred - q_true
