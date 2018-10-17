from __future__ import print_function, division

import numpy as np
from itertools import combinations_with_replacement
from gym.spaces import Box, Discrete


class ObservationDomainError(Exception):
    pass


class ActionDomainError(Exception):
    pass


class ObservationTypeError(Exception):
    pass


class ActionTypeError(Exception):
    pass


def check_observation(s, env):
    if not env.observation_space.contains(s):
        raise ObservationDomainError("Observation space {} does not contain observation {}".format(env.observation_space, s))


def check_action(a, env):
    if not env.action_space.contains(a):
        raise ActionDomainError("Action space {} does not contain action {}".format(env.action_space, a))


def check_observation_action_pair(s, a, env):
    check_observation(s, env)
    check_action(a, env)


def check_observation_space_type(space_type, env):
    if not isinstance(env.observation_space, Discrete):
        raise ActionTypeError("The environment's observation space is not of type: {}".format(space_type))


def check_action_space_type(space_type, env):
    if not isinstance(env.action_space, space_type):
        raise ObservationTypeError("The environment's action space is not of type: {}".format(space_type))


def softmax(arr, axis=0):
    arr = np.clip(-30, 30, arr)
    arr = np.exp(arr)
    arr /= arr.sum(axis=axis)
    return arr


def one_hot_vector(i, n):
    if not 0 <= i < n:
        raise ValueError("i must be a non-negative and smaller than n")
    x = np.zeros(int(n))
    x[int(i)] = 1.0
    return x


def feature_vector(env, s, a=None, intercept=True, polynomial_degree=1):
    """Translate state or state-action pair to feature vector.

    Args:
        env (gym-environment): This is needed to get the dimensions of the state-action space.
        s (gym-observation): The observation or 'state'.
        a (gym-action): The action.
        intercept (bool): Whether to add include an intercept in the set of weights.
        polynomial_degree (int): The degree for polynomial interactions, e.g. 2 means quadratic interactions.

    Returns:
        x (numpy-array): The feature vector.
    """
    check_observation(s, env)

    # ensure that the state is a 1d array
    if isinstance(env.observation_space, Discrete):
        x = one_hot_vector(s, env.observation_space.n)
    elif isinstance(env.observation_space, Box):
        x = np.ravel(s)
    else:
        raise NotImplementedError("Haven't yet implemented action spaces of type: {}".format(type(env.observation_space)))

    # add action to feature vector
    if a is not None:
        check_action(a, env)
        if isinstance(env.action_space, Discrete):
            a = one_hot_vector(a, env.action_space.n)
            x = np.hstack((x, a))
        elif isinstance(env.action_space, Box):
            a = np.ravel(a)
            x = np.hstack((x, a))
        else:
            raise NotImplementedError("Haven't yet implemented action spaces of type: {}".format(type(env.action_space)))

    # no fancy stuff? then we're done
    if not intercept and polynomial_degree == 1:
        return x

    # add intercept
    x = np.hstack(([1.0], x))

    # create polynomial interactions
    if polynomial_degree > 1:
        interactions = list(combinations_with_replacement(x, polynomial_degree))
        x = np.prod(interactions, axis=1)
        if not intercept:
            x = x[1:]

    return x
