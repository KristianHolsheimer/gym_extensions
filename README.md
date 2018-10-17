# Extensions to OpenAI gym

This repository is my own personal sandbox for playing with reinforcement learning algorithms. The reason why I started this is that I noticed that every time I wrote a little quick-n-dirty implementation of an RL algorithm, the code was so horrific and my results essentially irreproducible. So I figured that in order to progress, my code had to be more organized.


**Disclaimer:** Although the code in here should be reasonably clean and bug-free, it's by no means complete.


## Organisation of Code

The way I chose to organize the code is that I defined *value functions*, *policies* and *algorithms*. This is particularly useful for RL algorithms like Q-learning and I hope that other methods like policy-gradient methods will fit in this mold as well.


The central object *Q(s, a)* is defined as an abstract class `BaseQ`, which is refined in a variety of specific implementations, such as

- `TabularQ`
- `LinearQ`
- `NeuralNetQ` (not yet implemented)

The policy (or policies) derived from such Q-functions is implemented as a wrapper class:

- `PolicyQ`

In order to update the state-action value function, we use algorithm objects, such as

- `Sarsa`
- `QLearning`

There are also state-only value functions such as `LinearV`, which can be updated e.g. using the `ValueTD0` algorithm.


## A Simple (Working) Example:

```python
import gym
from gym_extensions.value_functions import LinearQ
from gym_extensions.algorithms import Sarsa
from gym_extensions.policies import PolicyQ

env = gym.make('CartPole-v0')
q = LinearQ(env, polynomial_degree=2)
p = PolicyQ(q)
algo = Sarsa(q, alpha=0.05, gamma=0.75)

# counter for early stopping
consecutive_successes = 0


for episode in range(1000):

    # init
    s = env.reset()
    a = env.action_space.sample()

    for t in range(200):

        # sample from environment
        s_next, r, done, _ = env.step(a)

        # use the PolicyQ object
        a_next = p.greedy(s_next)

        # if episode finished unsuccessfully, we'll hand in some of our return
        if done and t < 199:
            r = -5

        # update Q-function using SARSA algorithm
        algo.update(s, a, r, s_next, a_next)


        if done:
            if t == 199:
                consecutive_successes += 1
                print(f"episode={episode+1},  t={t+1}, consecutive_successes={consecutive_successes}")
            else:
                consecutive_successes = 0
                print(f"episode={episode+1},  t={t+1}, failed")
            break

        # prepare for next timestep
        s, a = s_next, a_next

    if consecutive_successes == 10:
        p.to_file('data/CartPole-v0.policy')
        break


env.close()
```

For a more elaborate examples, check out the [notebooks](./notebooks).
